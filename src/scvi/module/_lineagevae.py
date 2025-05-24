from __future__ import annotations

import logging
import torch
from torch.distributions import Distribution
import numpy as np
import scipy.sparse as sp

from ._vae import VAE
from scvi.nn import Encoder, MaskedLinearDecoder, VelocityDecoder
from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import (
    LossOutput,
    auto_move_data,
)
from scvi.utils import unsupported_if_adata_minified

import torch.nn.functional as F

logger = logging.getLogger(__name__)

class LINEAGEVAE(VAE):
    """Linear-decoded Variational auto-encoder model with masked linear decoder and kNN velocity loss."""

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "normal",
        use_batch_norm: bool = True,
        bias: bool = False,
        latent_distribution: str = "normal",
        use_observed_lib_size: bool = False,
        mask: torch.Tensor | None = None,
        input_layer: np.ndarray | None = None,
        K: int = 10,
        velocity_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers_encoder,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_observed_lib_size=use_observed_lib_size,
            **kwargs,
        )

        # register mask
        if mask is not None:
            self.register_buffer("mask", mask)

        # register global input matrix for kNN lookup
        if input_layer is None:
            raise ValueError("`input_layer` must be provided for kNN velocity loss.")
        
        if sp.issparse(input_layer):
            arr = input_layer.toarray()

        else:
            # if it’s already an ndarray we can use it, otherwise coerce
            import numpy as np
            arr = np.asarray(input_layer)

        arr = arr.astype(np.float32, copy=False)
        
        self.register_buffer(
            "full_data",
            torch.from_numpy(np.asarray(arr)).float(),
        )

        # kNN hyperparameters
        self.K = K
        self.velocity_loss_weight = velocity_loss_weight

        self.use_batch_norm = use_batch_norm
        # encoders
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=True,
            use_layer_norm=False,
            return_dist=True,
        )
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
            return_dist=True,
        )
        # masked linear decoder
        self.decoder = MaskedLinearDecoder(
            n_input=n_latent,
            n_output=n_input,
            mask=self.mask,
            n_cat_list=[n_batch],
            use_batch_norm=False,
            use_layer_norm=False,
            bias=bias,
        )

        # velocity decoder (simple FFN)
        self.velo_decoder = VelocityDecoder(
            n_input=n_latent,
            n_output=n_input,
            n_hidden=n_hidden,
            n_cat_list=[n_batch],
            use_batch_norm=True,
            use_layer_norm=False,
        )

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        # call parent for PX, PZ, PL
        outputs = super().generative(
            z, library, batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            size_factor=size_factor,
            y=y,
            transform_batch=transform_batch,
        )
        # add velocity prediction
        velo = self.velo_decoder(z)
        outputs[MODULE_KEYS.VELOCITY_KEY] = velo
        return outputs

    def _velocity_loss(
        self,
        velocity_pred: torch.Tensor,  # (B, G)
        x: torch.Tensor,              # (B, G)
        cell_idx: torch.Tensor,       # (B,)
    ) -> torch.Tensor:
        B, G = x.shape
        K = self.K
        # gather kNN indices buffer
        neigh_idx = self.nn_indices[cell_idx, :K]            # (B, K)
        flat_idx = neigh_idx.reshape(-1)                     # (B*K,)
        neigh_data = (
            self.full_data
                .index_select(0, flat_idx)
                .view(B, K, G)
        )
        # differences and cosine similarity
        diffs   = neigh_data - x.unsqueeze(1)                # (B, K, G)
        cos_sim = F.cosine_similarity(diffs, velocity_pred.unsqueeze(1), dim=-1)  # (B,K)
        max_sim, _ = cos_sim.max(dim=1)                      # (B,)
        return (1.0 - max_sim).mean()

    @unsupported_if_adata_minified
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: torch.Tensor | float = 1.0,
    ) -> LossOutput:
        base_out = super().loss(
            tensors, inference_outputs, generative_outputs, kl_weight
        )
        # velocity prediction from generative outputs
        vel = generative_outputs[MODULE_KEYS.VELOCITY_KEY]
        # raw counts and global index
        x = tensors[REGISTRY_KEYS.X_KEY]
        idxs = tensors[REGISTRY_KEYS.INDICES_KEY].squeeze(-1)
        # compute heuristic loss
        velo_loss = self._velocity_loss(vel, x, idxs)
        total = base_out.loss + self.velocity_loss_weight * velo_loss
        extra = dict(base_out.extra_metrics)
        extra["velocity_loss"] = velo_loss
        return LossOutput(
            loss=total,
            reconstruction_loss=base_out.reconstruction_loss,
            kl_local=base_out.kl_local,
            extra_metrics=extra,
        )

    @torch.inference_mode()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights in the linear decoder."""
        if self.use_batch_norm:
            # With batch norm: B W
            w = self.decoder.factor_regressor.fc_layers[0][0].weight
            bn = self.decoder.factor_regressor.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            loadings = torch.diag(b) @ w
        else:
            loadings = self.decoder.factor_regressor.fc_layers[0][0].weight
        loadings = loadings.detach().cpu().numpy()
        # if batches were concatenated in mask, slice them off
        if self.n_batch > 1:
            loadings = loadings[:, :-self.n_batch]
        return loadings
