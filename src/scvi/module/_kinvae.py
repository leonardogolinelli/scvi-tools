from __future__ import annotations

import logging
import torch
import numpy as np

from ._vae import VAE
from scvi.nn import Encoder, MaskedLinearDecoder

logger = logging.getLogger(__name__)


class KINVAE(VAE):
    """Linear-decoded Variational auto-encoder model with masked linear decoder."""

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
        gene_likelihood: str = "nb",
        use_batch_norm: bool = True,
        bias: bool = False,
        latent_distribution: str = "normal",
        use_observed_lib_size: bool = False,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        mask: binary tensor of shape (2 * n_input, n_latent) to zero decoder weights
        """
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

        # register mask buffer
        self.register_buffer("mask", mask)


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
            use_batch_norm=use_batch_norm,
            use_layer_norm=False,
            bias=bias,
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
