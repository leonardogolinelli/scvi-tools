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

if TYPE_CHECKING:
    from typing import Literal

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
        layer: str | None = None,             # keep track of which layer you registered as X_KEY
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
        x_field = self.adata_manager.data_registry[REGISTRY_KEYS.X_KEY]

        # 2) its attr_name is the user’s layer string (or None if they used adata.X)
        layer_name = x_field.attr_key

        # 3) fetch the matrix
        if layer_name is None:
            input_layer = adata.X
        else:
            input_layer = adata.layers[layer_name]

        # 4) now instantiate your module, forwarding the new arguments
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
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
            # ▶ NEW:
            input_layer=input_layer,
            K=K,
            **model_kwargs,
        )
        
        data_for_fit = adata.obsm["X_pca"]
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
        layer: str | None = None,
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
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_scvi_cell_index"),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

