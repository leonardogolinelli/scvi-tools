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
from torch.nn.functional import one_hot


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
        self.px_r = torch.nn.Parameter(torch.randn(2*n_input))



        self.use_batch_norm = use_batch_norm
        # encoders
        self.z_encoder = Encoder(
            2*n_input,
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
            2*n_input,
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

        u = tensors[REGISTRY_KEYS.UNSPLICED_KEY]
        s = tensors[REGISTRY_KEYS.SPLICED_KEY]
        x = torch.cat([u, s], dim=1)
        
        if loader == "full_data":
            return {
                MODULE_KEYS.X_KEY: x,
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
    def generative(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process."""
        from torch.nn.functional import linear

        from scvi.distributions import (
            NegativeBinomial,
            Normal,
            Poisson,
            ZeroInflatedNegativeBinomial,
        )

        # TODO: refactor forward function to not rely on y
        # Likelihood distribution
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_input = torch.cat([decoder_input, batch_rep], dim=-1)
            px_rate, px_scale = self.decoder(
                decoder_input,
                size_factor,
                *categorical_input,
                y,
            )
        else:
            px_rate, px_scale = self.decoder(
                decoder_input,
                size_factor,
                batch_index,
                *categorical_input,
                y,
            )

        if self.dispersion == "gene-label":
            px_r = linear(
                one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = linear(one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        print(f"px_rate shape: {px_rate.shape}, px_scale shape: {px_scale.shape}, px_r shape: {px_r.shape}")

        px = Normal(px_rate, px_r, normal_mu=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        # now you can use x however you like
        velo = self.velo_decoder(z, x)

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
            MODULE_KEYS.VELOCITY_KEY: velo,
        }
    

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
        kl_weight: torch.tensor | float = 1.0,
    ) -> LossOutput:
        if self.phase == 1:
            """Compute the loss."""
            from torch.distributions import kl_divergence

            u = tensors[REGISTRY_KEYS.UNSPLICED_KEY]
            s = tensors[REGISTRY_KEYS.SPLICED_KEY]
            x = torch.cat([u, s], dim=1)
            kl_divergence_z = kl_divergence(
                inference_outputs[MODULE_KEYS.QZ_KEY], generative_outputs[MODULE_KEYS.PZ_KEY]
            ).sum(dim=-1)
            if not self.use_observed_lib_size:
                kl_divergence_l = kl_divergence(
                    inference_outputs[MODULE_KEYS.QL_KEY], generative_outputs[MODULE_KEYS.PL_KEY]
                ).sum(dim=1)
            else:
                kl_divergence_l = torch.zeros_like(kl_divergence_z)
            print("ciao")
            print(x.shape)
            reconst_loss = -generative_outputs[MODULE_KEYS.PX_KEY].log_prob(x).sum(-1)

            kl_local_for_warmup = kl_divergence_z
            kl_local_no_warmup = kl_divergence_l

            weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

            loss = torch.mean(reconst_loss + weighted_kl_local)

            # a payload to be used during autotune
            if self.extra_payload_autotune:
                extra_metrics_payload = {
                    "z": inference_outputs["z"],
                    "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                    "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
                }
            else:
                extra_metrics_payload = {}

            return LossOutput(
                loss=loss,
                reconstruction_loss=reconst_loss,
                kl_local={
                    MODULE_KEYS.KL_L_KEY: kl_divergence_l,
                    MODULE_KEYS.KL_Z_KEY: kl_divergence_z,
                },
                extra_metrics=extra_metrics_payload,
            )
        
        else:
            vel = generative_outputs[MODULE_KEYS.VELOCITY_KEY]
            unspliced = tensors[REGISTRY_KEYS.UNSPLICED_KEY]
            spliced = tensors[REGISTRY_KEYS.SPLICED_KEY]
            u_s = torch.cat([unspliced, spliced], dim=1)  # (B, G*2)
            idx = tensors[REGISTRY_KEYS.INDICES_KEY].squeeze(-1)

            p_sign = self.velo_decoder.p_sign
            target = torch.full_like(p_sign, 0.25)
            loss_uniform = F.mse_loss(p_sign, target, reduction='mean')

            # compute just the velocity‐only loss
            velo_loss = self._velocity_loss(vel, u_s, idx) + 0.1*loss_uniform

            zeros = torch.zeros(unspliced.shape[0], device=velo_loss.device)

            # return a “pure” velocity LossOutput
            return LossOutput(
                loss=velo_loss,
                # zeros for all the ELBO bits so metrics see nothing
                reconstruction_loss=zeros,
                kl_local=zeros,
                extra_metrics={"velocity_loss": velo_loss, "loss_uniform": loss_uniform},
            )


    """def loss(
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
        )"""

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
