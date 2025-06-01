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

        velocity, velocity_u = np.split(velos, 2, axis=-1)
        
        return velocity, velocity_u
    
    @torch.inference_mode()
    def get_directional_uncertainty(
        self,
        n_samples: int = 50,
        gene_list: Iterable[str] = None,
        n_jobs: int = -1,
    ):
        adata = self._validate_anndata(self.adata)

        logger.info("Sampling from model...")
        velocities, _ = self.get_velocity(
            n_samples=n_samples, return_mean=False, gene_list=gene_list
        )  # (n_samples, n_cells, n_genes)

        df, cosine_sims = self._compute_directional_statistics_tensor(
            tensor=velocities, n_jobs=n_jobs, n_cells=adata.n_obs
        )
        df.index = adata.obs_names

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
    
    def compute_extrinisic_uncertainty(self, n_samples=25, n_jobs=-1) -> pd.DataFrame:
        import scvelo as scv
        from scvi.utils import track
        from contextlib import redirect_stdout
        import io

        extrapolated_cells_list = []
        for i in track(range(n_samples)):
            with io.StringIO() as buf, redirect_stdout(buf):
                vkey = "velocities_velovi_{i}".format(i=i)
                velocity, velocity_u = self.get_velocity(self.adata, n_samples=1, return_mean=True)
                self.adata.layers[vkey] = velocity
                scv.tl.velocity_graph(self.adata, vkey=vkey, sqrt_transform=False, approx=True)
                t_mat = scv.utils.get_transition_matrix(
                    self.adata, vkey=vkey, self_transitions=True, use_negative_cosines=True
                )
                extrapolated_cells = np.asarray(t_mat @ self.adata.layers["Ms"])
                extrapolated_cells_list.append(extrapolated_cells)
        extrapolated_cells = np.stack(extrapolated_cells_list)
        df, _ = self._compute_directional_statistics_tensor(extrapolated_cells, n_jobs=n_jobs, n_cells=self.adata.n_obs)
        return df

