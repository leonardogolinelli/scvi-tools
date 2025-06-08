from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, NumericalObsField
from scvi.model._utils import _init_library_size
from scvi.model.base import TwoPhaseTrainingMixin
from scvi.module._lineagevae import LINEAGEVAE
from scvi.utils import setup_anndata_dsp
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


from .base import BaseModelClass, RNASeqMixin, VAEMixin

from collections.abc import Iterator, Sequence
from joblib import Parallel, delayed

from torch import Tensor

from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

class LINEAGEVI(RNASeqMixin, VAEMixin, TwoPhaseTrainingMixin, BaseModelClass):
    _module_cls = LINEAGEVAE

    def __init__(
        self,
        adata: AnnData,
        K: int = 10,                           # ← NEW: how many neighbors to use
        distance_key : str = "distances",      # ← NEW: key to use for computing neighbor indices
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["normal"] = "normal",
        latent_distribution: Literal["normal", "ln"] = "normal",
        alpha: float = 0.1,
        **model_kwargs,
    ):
        super().__init__(adata)

        # 1) initialize library‐size prior
        n_batch = self.summary_stats.n_batch
        library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        # 2) build mask as before
        mask = torch.tensor(adata.varm["I"], dtype=torch.float32)
        mask = torch.cat([mask, mask], dim=0)

        # 1) grab the Field object under X_KEY
        unspliced_layer = adata.layers[REGISTRY_KEYS.UNSPLICED_KEY]
        spliced_layer = adata.layers[REGISTRY_KEYS.SPLICED_KEY]

        # 4) now instantiate your module, forwarding the new arguments
        self.module = self._module_cls(
            n_input=adata.shape[1],#self.summary_stats.n_vars,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            mask=mask,
            unspliced_layer=unspliced_layer,
            spliced_layer=spliced_layer,
            K=K,
            alpha=alpha,  # ← NEW: alpha for the KNN graph
            **model_kwargs,
        )
        
        data_for_fit = spliced_layer
        nbrs = NearestNeighbors(n_neighbors=K + 1, metric="euclidean")
        nbrs.fit(data_for_fit)
        _, all_idxs = nbrs.kneighbors(data_for_fit)

        # 3) slice off the self-index and keep only K neighbors
        nn_idx = all_idxs[:, 1 : K + 1]  # shape (n_cells, K)

        # 4) register on the module so it lands on GPU
        self.module.register_buffer("nn_indices", torch.from_numpy(nn_idx).long())

        self._model_summary_string = (
            f"LINEAGEVI Model with n_hidden={n_hidden}, n_latent={n_latent}, "
            f"n_layers={n_layers}, dropout_rate={dropout_rate}, dispersion={dispersion}, "
            f"gene_likelihood={gene_likelihood}, latent_distribution={latent_distribution}"
            f"LINEAGEVI w/ K={K}, distance_key={distance_key}, "
        )
        self.n_latent = n_latent
        self.init_params_ = self._get_init_params(locals())

    def get_loadings(self) -> pd.DataFrame:
        """Extract per-gene weights in the linear decoder.

        Shape is genes by `n_latent`.

        """
        cols = [f"Z_{i}" for i in range(self.n_latent)]
        var_names = self.adata.var_names
        loadings = pd.DataFrame(self.module.get_loadings(), index=var_names, columns=cols)

        return loadings


    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        labels_key: str | None = None,
        unspliced_key: str | None = None,
        spliced_key: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_layer)s
        """
        adata.obs["_scvi_cell_index"] = np.arange(adata.n_obs, dtype=int)
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.UNSPLICED_KEY, unspliced_key, is_count_data=False),
            LayerField(REGISTRY_KEYS.SPLICED_KEY, spliced_key, is_count_data=False),
            LayerField(REGISTRY_KEYS.X_KEY, spliced_key, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_scvi_cell_index"),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
    
    @torch.inference_mode()
    def get_velocity(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        gene_list: Sequence[str] | None = None,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_negative_velo: bool = True,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
    ) -> np.ndarray:
        """
        Estimate RNA velocity (cells × genes) by sampling from the posterior,
        using module.inference + module.generative directly.

        Parameters
        ----------
        adata
            AnnData to use (defaults to the one passed at initialization).
        indices
            Which cells to use (default: all).
        gene_list
            Return velocities only for this subset of genes.
        n_samples
            How many posterior samples per cell.
        batch_size
            Minibatch size (default scvi.settings.batch_size).
        return_mean
            If True, average over the n_samples per cell (→ [cells, genes]);
            else return all samples (→ [n_samples, cells, genes]).
        return_negative_velo
            If True, multiply all velocities by -1 before returning.
        dataloader
            You can pass your own iterator; otherwise one is built for you.

        Returns
        -------
        A NumPy array of shape
        - `(cells, G')` if `n_samples=1` or `return_mean=True`,
        - `(n_samples, cells, G')` otherwise,
        where `G' = len(gene_list)` if given, else `G = n_genes`.
        """
        import numpy as np
        import torch
        from scvi.data._utils import _validate_adata_dataloader_input
        from scvi.module._constants import MODULE_KEYS

        # 1) validate/train check
        self._check_if_trained(warn=False)
        _validate_adata_dataloader_input(self, adata, dataloader)

        # 2) build gene mask
        adata0 = self._validate_anndata(adata)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = list(adata0.var_names)
            gene_mask = [g in gene_list for g in all_genes]

        # 3) build or reuse dataloader
        if dataloader is None:
            if indices is None:
                indices = np.arange(adata0.n_obs)
            dataloader = self._make_data_loader(
                adata=adata0, indices=indices, batch_size=batch_size
            )
        else:
            for p in [indices, batch_size]:
                if p is not None:
                    Warning(f"Ignoring {p!r}; custom dataloader provided.")

        all_vels = []
        # 4) loop over minibatches
        for tensors in dataloader:
            samples: list[torch.Tensor] = []
            for _ in range(n_samples):
                inf_out = self.module.inference(
                    **self.module._get_inference_input(tensors)
                )
                gen_in = self.module._get_generative_input(tensors, inf_out)
                gen_out = self.module.generative(**gen_in)
                vel = gen_out[MODULE_KEYS.VELOCITY_KEY]  # (B, G)
                samples.append(vel.cpu())
            samp_tensor = torch.stack(samples, dim=0)  # (n_samples, B, G)

            if return_mean:
                batch_vel = samp_tensor.mean(dim=0)  # (B, G)
            else:
                batch_vel = samp_tensor         # (n_samples, B, G)

            # subset genes if requested
            if isinstance(gene_mask, list):
                if batch_vel.ndim == 3:
                    batch_vel = batch_vel[..., gene_mask]
                else:
                    batch_vel = batch_vel[:, gene_mask]

            if return_negative_velo:
                batch_vel.neg_()

            all_vels.append(batch_vel)

        # 5) stitch batches back together
        first = all_vels[0]
        if first.ndim == 3:
            final = torch.cat(all_vels, dim=1)  # (n_samples, total_cells, G')
        else:
            final = torch.cat(all_vels, dim=0)  # (total_cells, G')

        velos = final.numpy()

        velocity_u, velocity = np.split(velos, 2, axis=-1)
        
        return velocity_u, velocity
    
    @torch.inference_mode()
    def get_directional_uncertainty(
        self,
        n_samples: int = 50,
        gene_list: Iterable[str] = None,
        n_jobs: int = -1,
        show_plot: bool = True,
    ):
        import scanpy as sc
        adata = self._validate_anndata(self.adata)

        logger.info("Sampling from model...")
        velocity_u, velocity = self.get_velocity(
            n_samples=n_samples, return_mean=False, gene_list=gene_list
        )  # (n_samples, n_cells, n_genes)

        df, cosine_sims = self._compute_directional_statistics_tensor(
            tensor=velocity, n_jobs=n_jobs, n_cells=adata.n_obs
        )
        df.index = adata.obs_names

        for c in df.columns:
            print(f'Adding {c} to adata.obs')
            adata.obs[c] = np.log10(df[c].values) 

        if show_plot:
            print('Plotting directional_cosine_sim_variance')
            sc.pl.umap(
                adata, 
                color="directional_cosine_sim_variance",
                vmin="p1",
                vmax="p99",
            )

        return df, cosine_sims

    def _compute_directional_statistics_tensor(
        self, tensor: np.ndarray, n_jobs: int, n_cells: int
    ) -> pd.DataFrame:
        df = pd.DataFrame(index=np.arange(n_cells))
        df["directional_variance"] = np.nan
        df["directional_difference"] = np.nan
        df["directional_cosine_sim_variance"] = np.nan
        df["directional_cosine_sim_difference"] = np.nan
        df["directional_cosine_sim_mean"] = np.nan
        logger.info("Computing the uncertainties...")
        results = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(self._directional_statistics_per_cell)(tensor[:, cell_index, :])
            for cell_index in range(n_cells)
        )
        # cells by samples
        cosine_sims = np.stack([results[i][0] for i in range(n_cells)])
        df.loc[:, "directional_cosine_sim_variance"] = [
            results[i][1] for i in range(n_cells)
        ]
        df.loc[:, "directional_cosine_sim_difference"] = [
            results[i][2] for i in range(n_cells)
        ]
        df.loc[:, "directional_variance"] = [results[i][3] for i in range(n_cells)]
        df.loc[:, "directional_difference"] = [results[i][4] for i in range(n_cells)]
        df.loc[:, "directional_cosine_sim_mean"] = [results[i][5] for i in range(n_cells)]

        return df, cosine_sims
    
    def _directional_statistics_per_cell(
        self,
        tensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Internal function for parallelization.

        Parameters
        ----------
        tensor
            Shape of samples by genes for a given cell.
        """
        n_samples = tensor.shape[0]
        # over samples axis
        mean_velocity_of_cell = tensor.mean(0)
        cosine_sims = [
            self._cosine_sim(tensor[i, :], mean_velocity_of_cell) for i in range(n_samples)
        ]
        angle_samples = [np.arccos(el) for el in cosine_sims]
        return (
            cosine_sims,
            np.var(cosine_sims),
            np.percentile(cosine_sims, 95) - np.percentile(cosine_sims, 5),
            np.var(angle_samples),
            np.percentile(angle_samples, 95) - np.percentile(angle_samples, 5),
            np.mean(cosine_sims),
        )
    
    def _centered_unit_vector(self, vector: np.ndarray) -> np.ndarray:
        """Returns the centered unit vector of the vector."""
        vector = vector - np.mean(vector)
        return vector / np.linalg.norm(vector)

    def _cosine_sim(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Returns cosine similarity of the vectors."""
        v1_u = self._centered_unit_vector(v1)
        v2_u = self._centered_unit_vector(v2)
        return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    
    def compute_extrinisic_uncertainty(
            self,
            n_samples=25, 
            n_jobs=-1,
            show_plot=True
        ) -> pd.DataFrame:

        import scanpy as sc
        import scvelo as scv
        from scvi.utils import track
        from contextlib import redirect_stdout
        import io

        adata = self._validate_anndata(self.adata)

        extrapolated_cells_list = []
        for i in track(range(n_samples)):
            with io.StringIO() as buf, redirect_stdout(buf):
                vkey = "velocities_velovi_{i}".format(i=i)
                velocity_u, velocity = self.get_velocity(adata, n_samples=1, return_mean=True)
                adata.layers[vkey] = velocity
                scv.tl.velocity_graph(adata, vkey=vkey, sqrt_transform=False, approx=True)
                t_mat = scv.utils.get_transition_matrix(
                    adata, vkey=vkey, self_transitions=True, use_negative_cosines=True
                )
                extrapolated_cells = np.asarray(t_mat @ adata.layers["Ms"])
                extrapolated_cells_list.append(extrapolated_cells)
        extrapolated_cells = np.stack(extrapolated_cells_list)
        df, _ = self._compute_directional_statistics_tensor(extrapolated_cells, n_jobs=n_jobs, n_cells=self.adata.n_obs)

        for c in df.columns:
            adata.obs[c + "_extrinisic"] = np.log10(df[c].values)

        if show_plot:
            sc.pl.umap(
                adata, 
                color="directional_cosine_sim_variance_extrinisic",
                vmin="p1", 
                vmax="p99", 
            )

        return df

    @torch.inference_mode()
    def latent_directions(self, method="sum", get_confidence=False,
                          key_added='directions'):
        """Get directions of upregulation for each latent dimension.
           Multipling this by raw latent scores ensures positive latent scores correspond to upregulation.

           Parameters
           ----------
           method: String
                Method of calculation, it should be 'sum' or 'counts'.
           get_confidence: Boolean
                Only for method='counts'. If 'True', also calculate confidence
                of the directions.
           adata: AnnData
                An AnnData object to store dimensions. If 'None', self.adata is used.
           key_added: String
                key of adata.uns where to put the dimensions.
        """

        terms_weights = self.module.decoder.linear.weight  # (n_out, n_latent)

        if method == "sum":
            signs = terms_weights.sum(0).cpu().numpy()
            signs[signs>0] = 1.
            signs[signs<0] = -1.
            confidence = None
        elif method == "counts":
            # 1) count nonzero and upregulated weights as torch tensors
            num_nz_t    = torch.count_nonzero(terms_weights,        dim=0)
            upreg_genes_t = torch.count_nonzero(terms_weights > 0, dim=0)

            # 2) bring them to CPU and to NumPy
            num_nz       = num_nz_t.cpu().numpy()
            upreg_genes  = upreg_genes_t.cpu().numpy()

            # 3) compute the raw fraction
            prop = upreg_genes / (num_nz + (num_nz == 0))

            # 4) make signs and confidence arrays
            signs      = prop.copy()
            confidence = np.abs(prop - 0.5) / 0.5

            # 5) build a pure-NumPy mask for zero-count dimensions
            zero_mask = (num_nz == 0)

            # 6) apply that mask
            confidence[zero_mask] = 0

            # 7) threshold your signs
            signs[prop > 0.5] =  1.0
            signs[prop < 0.5] = -1.0
            signs[prop == 0.5] = 0.0
            signs[zero_mask]  = 0.0

        else:
            raise ValueError("Unrecognized method for getting the latent direction.")

        self.adata.uns[key_added] = signs
        if get_confidence and confidence is not None:
            self.adata.uns[key_added + '_confindence'] = confidence

    def latent_enrich(
        self,
        groups,
        comparison='rest',
        use_directions=False,
        directions_key='directions',
        select_terms=None,
        exact=True,
        key_added='bf_scores',
        active_gps=True,
    ):
        """Gene set enrichment test for the latent space. Test the hypothesis that latent scores
           for each term in one group (z_1) is bigger than in the other group (z_2).

           Puts results to `adata.uns[key_added]`. Results are a dictionary with
           `p_h0` - probability that z_1 > z_2, `p_h1 = 1-p_h0` and `bf` - bayes factors equal to `log(p_h0/p_h1)`.

           Parameters
           ----------
           groups: String or Dict
                A string with the key in `adata.obs` to look for categories or a dictionary
                with categories as keys and lists of cell names as values.
           comparison: String
                The category name to compare against. If 'rest', then compares each category against all others.
           n_sample: Integer
                Number of random samples to draw for each category.
           use_directions: Boolean
                If 'True', multiplies the latent scores by directions in `adata`.
           directions_key: String
                The key in `adata.uns` for directions.
           select_terms: Array
                If not 'None', then an index of terms to select for the test. Only does the test
                for these terms.
           exact: Boolean
                Use exact probabilities for comparisons.
           key_added: String
                key of adata.uns where to put the results of the test.
        """

        if isinstance(groups, str):
            cats_col = self.adata.obs[groups]
            cats = cats_col.unique()
        elif isinstance(groups, dict):
            cats = []
            all_cells = []
            for group, cells in groups.items():
                cats.append(group)
                all_cells += cells
            self.adata = self.adata[all_cells]
            cats_col = pd.Series(index=self.adata.obs_names, dtype=str)
            for group, cells in groups.items():
                cats_col[cells] = group
        else:
            raise ValueError("groups should be a string or a dict.")

        if comparison != "rest" and isinstance(comparison, str):
            comparison = [comparison]

        if comparison != "rest" and not set(comparison).issubset(cats):
            raise ValueError("comparison should be 'rest' or among the passed groups")

        scores = {}

        for cat in cats:
            if cat in comparison:
                continue

            cat_mask = cats_col == cat
            if comparison == "rest":
                others_mask = ~cat_mask
            else:
                others_mask = cats_col.isin(comparison)

            # Get all indices of the full AnnData object
            all_indices = np.arange(self.adata.n_obs)

            # Apply the boolean masks to get the relevant indices
            cat_indices = all_indices[cat_mask]
            others_indices = all_indices[others_mask]

            n_sample = min(len(cat_indices), len(others_indices))

            # Sample from the indices
            #n_sample = len(self.adata)
            choice_1 = np.random.choice(len(cat_indices), n_sample, replace=False)
            choice_2 = np.random.choice(len(others_indices), n_sample, replace=False)

            # Get the actual indices (not names) in self.adata
            z0_obs_indices = cat_indices[choice_1]
            z1_obs_indices = others_indices[choice_2]

            if use_directions:
                directions = self.adata.uns[directions_key]
            else:
                directions = None

            z = self.get_latent_representation(active_gps=active_gps, return_dist=exact)

            if not exact:
                z0 = z[z0_obs_indices, :]
                z1 = z[z1_obs_indices, :]
                if directions is not None:
                    z0 *= directions
                    z1 *= directions

                if select_terms is not None:
                    z0 = z0[:, select_terms]
                    z1 = z1[:, select_terms]

                to_reduce = z0 > z1

                zeros_mask = (np.abs(z0).sum(0) == 0) | (np.abs(z1).sum(0) == 0)
            else:
                from scipy.special import erfc

                means0 = z[0][z0_obs_indices, :]
                vars0 = z[1][z0_obs_indices, :]
                means1 = z[0][z1_obs_indices, :]
                vars1 = z[1][z1_obs_indices, :]

                if directions is not None:
                    means0 *= directions
                    means1 *= directions

                if select_terms is not None:
                    means0 = means0[:, select_terms]
                    means1 = means1[:, select_terms]
                    vars0 = vars0[:, select_terms]
                    vars1 = vars1[:, select_terms]

                to_reduce = (means1 - means0) / np.sqrt(2 * (vars0 + vars1))
                to_reduce = 0.5 * erfc(to_reduce)

                zeros_mask = (np.abs(means0).sum(0) == 0) | (np.abs(means1).sum(0) == 0)

            p_h0 = np.mean(to_reduce, axis=0)
            p_h1 = 1.0 - p_h0
            epsilon = 1e-12
            bf = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)

            p_h0[zeros_mask] = 0
            p_h1[zeros_mask] = 0
            bf[zeros_mask] = 0

            scores[cat] = dict(p_h0=p_h0, p_h1=p_h1, bf=bf)

        self.adata.uns[key_added] = scores
    

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata=None,
        indices=None,
        give_mean=True,
        mc_samples: int = 5_000,
        batch_size: int | None = None,
        return_dist: bool = False,
        dataloader: Iterator[dict[str, Tensor | None]] = None,
        active_gps: bool = False,
    ):
        # … (same as before up through collecting all_means/all_vars or all_z) …

        if return_dist:
            all_means, all_vars = super().get_latent_representation(
                adata=adata,
                indices=indices,
                give_mean=give_mean,
                mc_samples=mc_samples,
                batch_size=batch_size,
                return_dist=True,
                dataloader=dataloader,
            )
            if not active_gps:
                return all_means, all_vars

            # ─────────── re‑apply mask before computing norms ───────────
            W = self.module.decoder.linear.weight       # shape: (n_out, n_latent)
            W_masked = W * self.module.decoder.mask     # ensure masked entries are zero
            col_norms = W_masked.norm(dim=0).cpu().numpy()  # (n_latent,)
            active_mask = col_norms > 0.0
            # ──────────────────────────────────────────────────────────────

            active_indices = list(np.nonzero(active_mask)[0])
            all_names = [f"GP{i}" for i in range(W.shape[1])]
            active_names = [all_names[i] for i in active_indices]

            if adata is not None:
                adata.uns["active_gp_indices"] = active_indices
                adata.uns["active_gp_names"] = active_names
                print(
                    "Stored active GP indices in adata.uns['active_gp_indices'] "
                    "and names in adata.uns['active_gp_names']"
                )

            active_means = all_means[:, active_mask]
            active_vars = all_vars[:, active_mask]
            return active_means, active_vars

        else:
            all_z = super().get_latent_representation(
                adata=adata,
                indices=indices,
                give_mean=give_mean,
                mc_samples=mc_samples,
                batch_size=batch_size,
                return_dist=False,
                dataloader=dataloader,
            )
            if not active_gps:
                return all_z

            # ─────────── re‑apply mask before computing norms ───────────
            W = self.module.decoder.linear.weight       # shape: (n_out, n_latent)
            W_masked = W * self.module.decoder.mask     # ensure masked entries are zero
            col_norms = W_masked.norm(dim=0).cpu().numpy()  # (n_latent,)
            active_mask = col_norms > 0.0
            # ──────────────────────────────────────────────────────────────

            active_indices = list(np.nonzero(active_mask)[0])
            all_names = [f"GP{i}" for i in range(W.shape[1])]
            active_names = [all_names[i] for i in active_indices]

            if adata is not None:
                adata.uns["active_gp_indices"] = active_indices
                adata.uns["active_gp_names"] = active_names
                print(
                    "Stored active GP indices in adata.uns['active_gp_indices'] "
                    "and names in adata.uns['active_gp_names']"
                )

            return all_z[:, active_mask]
        
    def plot_top_gps_activation(self, latent_key="X_cvae", terms_key="terms", n=10):

        adata = self.adata

        latent_means = adata.obsm[latent_key].mean(0)
        sorted_idxs = np.argsort(np.abs(latent_means))[::-1][:n]

        # Retrieve corresponding term names
        gp_names = np.array(adata.uns[terms_key])[sorted_idxs]
        activations = latent_means[sorted_idxs]
        colors = ['blue' if val > 0 else 'red' for val in activations]

        plt.figure(figsize=(10, 6))
        plt.barh(gp_names[::-1], activations[::-1], color=colors[::-1])  # flip for top-down
        plt.xlabel('Activation')
        plt.title('Top {} Gene Programs by Absolute Activation'.format(n))
        plt.tight_layout()
        plt.show()


    def plot_top_gps_per_celltype(self,
                                groupby="cell_type", 
                                latent_key="X_cvae",
                                term_key="terms",
                                n=10,
                                target_group=None):
        """
        Plot barplots of top absolute gene program activations for a specific or all cell types.
        
        Parameters:
        - adata: AnnData object
        - groupby: Column in adata.obs to group cells by (e.g., "cell_type")
        - latent_key: Key in adata.obsm where gene program activations are stored
        - term_key: Key in adata.uns with gene program names
        - n: Number of top absolute gene programs to show per group
        - target_group: If specified, only plot this specific group (e.g., a single cell type)
        """

        adata = self.adata

        groups = [target_group] if target_group else adata.obs[groupby].unique()
        gp_names = np.array(adata.uns[term_key])

        for group in groups:
            idx = adata.obs[groupby] == group
            group_activations = adata.obsm[latent_key][idx].mean(axis=0)
            
            top_idx = np.argsort(np.abs(group_activations))[::-1][:n]
            top_gps = gp_names[top_idx]
            top_vals = group_activations[top_idx]
            colors = ['blue' if val > 0 else 'red' for val in top_vals]

            plt.figure(figsize=(10, 6))
            plt.barh(top_gps[::-1], top_vals[::-1], color=colors[::-1])  # reverse for descending top-to-bottom
            plt.xlabel("Activation")
            plt.ylabel("Gene Programs")
            plt.title(f"Top {n} Absolute Gene Programs in {group}")
            plt.tight_layout()
            plt.show()


    def scatter_terms(self,
                    term_x, 
                    term_y, 
                    latent_key="X_cvae", 
                    term_key="terms", 
                    groupby="clusters",
                    s=10,
                    alpha=0.8):
        """
        Scatter plot of cells in space of two gene programs using matplotlib,
        respecting Scanpy's color-to-group mapping.
        """

        adata = self.adata
        
        gp_names = list(adata.uns[term_key])
        try:
            idx_x = gp_names.index(term_x)
            idx_y = gp_names.index(term_y)
        except ValueError as e:
            raise ValueError(f"Term not found in {term_key}: {e}")

        X = adata.obsm[latent_key][:, [idx_x, idx_y]]
        groups = adata.obs[groupby]

        # Use categorical order if available
        if pd.api.types.is_categorical_dtype(groups):
            group_order = list(groups.cat.categories)
        else:
            group_order = sorted(groups.unique())

        if f"{groupby}_colors" in adata.uns:
            color_list = adata.uns[f"{groupby}_colors"]
            if len(color_list) != len(group_order):
                raise ValueError("Mismatch between number of colors and number of categories.")
            color_map = dict(zip(group_order, color_list))
        else:
            raise ValueError(f"Expected colors in adata.uns['{groupby}_colors']")

        plt.figure(figsize=(8, 6))
        for group in group_order:
            idx = groups == group
            plt.scatter(X[idx, 0], X[idx, 1], 
                        c=color_map[group], 
                        label=group, 
                        s=s, alpha=alpha, edgecolors='none')

        plt.xlabel(term_x)
        plt.ylabel(term_y)
        plt.title(f"{term_x} vs {term_y}")
        plt.legend(title=groupby, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()


    def plot_abs_bfs_key(self, scores, terms, key, n_points=30, lim_val=2.3, fontsize=8, scale_y=2, yt_step=0.3,
                     title=None, ax=None):
        txt_args = dict(
            rotation='vertical',
            verticalalignment='bottom',
            horizontalalignment='center',
            fontsize=fontsize,
        )

        ax = ax if ax is not None else plt.axes()
        ax.grid(False)

        bfs = np.abs(scores[key]['bf'])
        srt = np.argsort(bfs)[::-1][:n_points]
        top = bfs.max()

        ax.set_ylim(top=top * scale_y)
        yt = np.arange(0, top * 1.1, yt_step)
        ax.set_yticks(yt)

        ax.set_xlim(0.1, n_points + 0.9)
        xt = np.arange(0, n_points + 1, 5)
        xt[0] = 1
        ax.set_xticks(xt)

        for i, (bf, term) in enumerate(zip(bfs[srt], terms[srt])):
            ax.text(i+1, bf, term, **txt_args)

        ax.axhline(y=lim_val, color='red', linestyle='--', label='')

        ax.set_xlabel("Rank")
        ax.set_ylabel("Absolute log bayes factors")
        ax.set_title(key if title is None else title)

        return ax.figure

    def plot_abs_bfs(self, scores_key="bf_scores", terms: Union[str, list]="terms",
                    keys=None, n_cols=3, **kwargs):
        """\
        Plot the absolute bayes scores rankings.
        """

        from itertools import product

        adata = self.adata
        scores = adata.uns[scores_key]

        if isinstance(terms, str):
            terms = np.asarray(adata.uns[terms])
        else:
            terms = np.asarray(terms)

        if len(terms) != len(next(iter(scores.values()))["bf"]):
            raise ValueError('Incorrect length of terms.')

        if keys is None:
            keys = list(scores.keys())

        if len(keys) == 1:
            keys = keys[0]

        if isinstance(keys, str):
            return self.plot_abs_bfs_key(scores, terms, keys, **kwargs)

        n_keys = len(keys)

        if n_keys <= n_cols:
            n_cols = n_keys
            n_rows = 1
        else:
            n_rows = int(np.ceil(n_keys / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols)
        for key, ix in zip(keys, product(range(n_rows), range(n_cols))):
            if n_rows == 1:
                ix = ix[1]
            elif n_cols == 1:
                ix = ix[0]
            self.plot_abs_bfs_key(scores, terms, key, ax=axs[ix], **kwargs)

        n_inactive = n_rows * n_cols - n_keys
        if n_inactive > 0:
            for i in range(n_inactive):
                axs[n_rows-1, -(i+1)].axis('off')

        return fig

    def add_annotations(self, files, min_genes=0, max_genes=None, varm_key='I', uns_key='terms',
                    clean=True, genes_use_upper=True):
        """\
        Add annotations to an AnnData object from files.

        Parameters
        ----------
        adata
            Annotated data matrix.
        files
            Paths to text files with annotations. The function considers rows to be gene sets
            with name of a gene set in the first column followed by names of genes.
        min_genes
            Only include gene sets which have the total number of genes in adata
            greater than this value.
        max_genes
            Only include gene sets which have the total number of genes in adata
            less than this value.
        varm_key
            Store the binary array I of size n_vars x number of annotated terms in files
            in `adata.varm[varm_key]`. if I[i,j]=1 then the gene i is present in the annotation j.
        uns_key
            Sore gene sets' names in `adata.uns[uns_key]`.
        clean
            If 'True', removes the word before the first underscore for each term name (like 'REACTOME_')
            and cuts the name to the first thirty symbols.
        genes_use_upper
            if 'True', converts genes' names from files and adata to uppercase for comparison.
        """

        adata = self.adata
        
        files = [files] if isinstance(files, str) else files
        annot = []

        for file in files:
            with open(file) as f:
                p_f = [l.upper() for l in f] if genes_use_upper else f
                terms = [l.strip('\n').split() for l in p_f]

            if clean:
                terms = [[term[0].split('_', 1)[-1][:30]]+term[1:] for term in terms if term]
            annot+=terms

        var_names = adata.var_names.str.upper() if genes_use_upper else adata.var_names
        I = [[int(gene in term) for term in annot] for gene in var_names]
        I = np.asarray(I, dtype='int32')

        mask = I.sum(0) > min_genes
        if max_genes is not None:
            mask &= I.sum(0) < max_genes
        I = I[:, mask]
        adata.varm[varm_key] = I
        adata.uns[uns_key] = [term[0] for i, term in enumerate(annot) if i not in np.where(~mask)[0]]



