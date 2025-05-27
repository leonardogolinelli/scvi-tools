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
from scvi.data._constants import ADATA_MINIFY_TYPE
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
        unspliced_layer: np.ndarray | None = None,
        spliced_layer: np.ndarray | None = None,
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

        unspliced = unspliced_layer.toarray() if sp.issparse(unspliced_layer) else unspliced_layer
        spliced = spliced_layer.toarray() if sp.issparse(spliced_layer) else spliced_layer

        unspliced = unspliced.astype(np.float32, copy=False)
        spliced = spliced.astype(np.float32, copy=False)

        # concatenate full unspliced and spliced
        full_u_s = np.concatenate(
            [unspliced, spliced], axis=1
        ) # (B, G*2)

        self.register_buffer(
            "full_data",
            torch.from_numpy(np.asarray(full_u_s)).float(),
        )

        # kNN hyperparameters
        self.K = K
        self.velocity_loss_weight = velocity_loss_weight
        self.phase = 1

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
            n_output=2*n_input, # unspliced (mean,std), spliced (mean,std)
            mask=self.mask,
            n_cat_list=[n_batch],
            use_batch_norm=False,
            use_layer_norm=False,
            bias=bias,
        )

        # velocity decoder (simple FFN)
        self.velo_decoder = VelocityDecoder(
            n_input=n_latent,
            n_output=3*n_input, # alpha, beta, gamma for each gene
            n_hidden=n_hidden,
            n_cat_list=[n_batch],
            use_batch_norm=False,
            use_layer_norm=True,
            use_activation=True,
            dropout_rate=0
        )

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        if full_forward_pass or self.minified_data_type is None:
            loader = "full_data"
        elif self.minified_data_type in [
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR,
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR_WITH_COUNTS,
        ]:
            loader = "minified_data"
        else:
            raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")

        if loader == "full_data":
            return {
                MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.SPLICED_KEY],
                MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
                MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
                MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            }
        else:
            return {
                MODULE_KEYS.QZM_KEY: tensors[REGISTRY_KEYS.LATENT_QZM_KEY],
                MODULE_KEYS.QZV_KEY: tensors[REGISTRY_KEYS.LATENT_QZV_KEY],
                REGISTRY_KEYS.OBSERVED_LIB_SIZE: tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE],
            }    

    def _get_generative_input(self, tensors, inference_outputs):
        # first get the standard args
        gen_inputs = super()._get_generative_input(tensors, inference_outputs)
        # now build the x you actually want
        u = tensors[REGISTRY_KEYS.UNSPLICED_KEY]
        s = tensors[REGISTRY_KEYS.SPLICED_KEY]
        x = torch.cat([u, s], dim=1)
        # and inject it under the name your generative() expects:
        gen_inputs[MODULE_KEYS.X_KEY] = x
        return gen_inputs
    
    @auto_move_data
    def generative(self,
                   x,           # <-- will now get the x you just inserted
                   z,
                   library,
                   batch_index,
                   cont_covs=None,
                   cat_covs=None,
                   size_factor=None,
                   y=None,
                   transform_batch=None,
    ):
        # call the parent to get everything but 'x'
        outputs = super().generative(
            z,
            library,
            batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            size_factor=size_factor,
            y=y,
            transform_batch=transform_batch,
        )
        # now you can use x however you like
        velo = self.velo_decoder(z, x)
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

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: torch.Tensor | float = 1.0,
    ) -> LossOutput:
        # -------------------------------------------------------------------
        # PHASE 1: only the base VAE loss (reconstruction + KL)
        # PHASE 2: only the velocity loss
        # -------------------------------------------------------------------
        if self.phase == 1:
            # exactly as before, ignore velocity
            return super().loss(tensors, inference_outputs, generative_outputs, kl_weight)

        # phase == 2 → zero out the base and only apply velocity
        # pull out your predicted velocity + inputs
        
        vel = generative_outputs[MODULE_KEYS.VELOCITY_KEY]
        unspliced = tensors[REGISTRY_KEYS.UNSPLICED_KEY]
        spliced = tensors[REGISTRY_KEYS.SPLICED_KEY]
        u_s = torch.cat([unspliced, spliced], dim=1)  # (B, G*2)
        idx = tensors[REGISTRY_KEYS.INDICES_KEY].squeeze(-1)

        # compute just the velocity‐only loss
        velo_loss = self._velocity_loss(vel, u_s, idx)

        zeros = torch.zeros(unspliced.shape[0], device=velo_loss.device)

        # return a “pure” velocity LossOutput
        return LossOutput(
            loss=velo_loss,
            # zeros for all the ELBO bits so metrics see nothing
            reconstruction_loss=zeros,
            kl_local=zeros,
            extra_metrics={"velocity_loss": velo_loss},
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
