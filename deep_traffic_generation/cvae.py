# fmt: off
# fixing posix path error for windows
# check if current os is windows
import os

from deep_traffic_generation.core.networks import CN
if os.name == 'nt':
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from base64 import decode
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from deep_traffic_generation.core import VAE, cli_main
from deep_traffic_generation.core.datasets import DatasetParams, TrafficDataset
from deep_traffic_generation.core.lsr import VampPriorLSR, NormalLSR, factorized_GMM_LSR, factorized_VampPriorLSR
from response.responses import Autoencoder

class ConvDecoder(nn.Module):
    """Deconvolutional Network for mapping encoded representations back to the original input space.

    Args:
        encoding_dim: Dimensionality of the encoded representation.
        output_channels: Number of output features (output dimension, typically the input dimension of the encoder).
        seq_len: Original sequence length of the input.
        h_dims: List of dimensions for each deconvolutional/transposed convolutional layer.
        kernel_size: Kernel size for each deconvolutional layer.
        sampling_factor: Factor by which the sequence length was reduced in the encoder.
    """
    
    def __init__(
        self,
        encoding_dim: int,
        output_channels: int,
        seq_len: int,
        h_dims: List[int],
        kernel_size: int,
        sampling_factor: int,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.sampling_factor = sampling_factor
        self.h_dims = h_dims
        self.kernel_size = kernel_size

        # Calculate the dimension before flattening in the encoder
        h_dim = h_dims[-1] * (seq_len // sampling_factor)

        self.initial_layer = nn.Sequential(
            nn.Linear(encoding_dim, h_dim),
            nn.ReLU(inplace=True)
        )

        deconv_layers = []
        deconv_layers.append(nn.Upsample(scale_factor=sampling_factor))
        for i in range(len(h_dims)-1, 0, -1):
            deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=h_dims[i],
                        out_channels=h_dims[i-1],
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size-1)//2,
                    ),
                    nn.ReLU(inplace=True)
                )
            )
                

        deconv_layers.append(
            nn.ConvTranspose1d(
                in_channels=h_dims[0],
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size-1)//2,
            )
        )

        self.deconv_layers = nn.Sequential(*deconv_layers)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.initial_layer(x)
        x = x.view(-1, self.h_dims[-1], self.seq_len // self.sampling_factor)
        x = self.deconv_layers(x)
        return x

class CVAE(VAE, Autoencoder):
    """
    Convolutional Variational Autoencoder
    """

    _required_hparams = VAE._required_hparams + [
        "sampling_factor",
        "kernel_size"
    ]

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)
        self.example_input_array = torch.rand(
            (
                1,
                self.dataset_params["input_dim"],
                self.dataset_params["seq_len"],
            )
        )

        self.encoder = nn.Sequential(
            CN(
                in_channels=self.dataset_params["input_dim"],
                out_dim=self.hparams.h_dims[-1],
                channels=self.hparams.h_dims[:-1],
                kernel_size=self.hparams.kernel_size,
                h_activ=nn.ReLU(inplace=True),
                dropout=self.hparams.dropout,
                # batch_norm=True,
            ),
            nn.AvgPool1d(self.hparams.sampling_factor),
            nn.Flatten(),
        )

        h_dim = self.hparams.h_dims[-1] * (
            int(self.dataset_params["seq_len"] / self.hparams.sampling_factor)
        )

        if self.hparams.prior == "vampprior":
            self.lsr = VampPriorLSR(
                original_dim=self.dataset_params["input_dim"],
                original_seq_len=self.dataset_params["seq_len"],
                input_dim=h_dim,
                out_dim=self.hparams.encoding_dim,
                encoder=self.encoder,
                n_components=self.hparams.n_components,
            )

        elif self.hparams.prior == "standard":
            self.lsr = NormalLSR(
                input_dim=h_dim,
                out_dim=self.hparams.encoding_dim,
            )

        else:
            raise Exception("Wrong name of the prior!")

        self.decoder = ConvDecoder(
            encoding_dim=self.hparams.encoding_dim,
            output_channels=self.dataset_params["input_dim"],
            seq_len=self.dataset_params["seq_len"],
            # h_dims=list(reversed(self.hparams.h_dims[:-1])) + [self.hparams.h_dims[-1]],  # Ensure the dimensions align with the encoder's reversed process
            h_dims=self.hparams.h_dims,
            kernel_size=self.hparams.kernel_size,
            sampling_factor=self.hparams.sampling_factor,
        )

        # non-linear activation after decoder
        self.out_activ = nn.Identity()
        # self.out_activ = nn.Tanh()

        self.latent_dim = self.hparams.encoding_dim

    def test_step(self, batch, batch_idx):
        x, info = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", np.log(loss.cpu().item()), sync_dist=True)
        return torch.transpose(x, 1, 2), torch.transpose(x_hat, 1, 2), info

    @classmethod
    def network_name(cls) -> str:
        return "cvae"

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds CVAE arguments to ArgumentParser.

        List of arguments:

            * ``--kernel``: Size of the kernel to use in Temporal Convolutional
              layers. Default to :math:`16`.
            * ``--prior``: choice of the prior (standard or vampprior). Default to
            "vampprior".
            * ``--n_components``: Number of components for the Gaussian Mixture
              modelling the prior. Default to :math:`300`.
            * ``--sampling_factor``: Sampling factor to reduce the sequence
              length after Temporal Convolutional layers. Default to
              :math:`10`.

        .. note::
            It adds also the argument of the inherited class `VAE`.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--sampling_factor",
            dest="sampling_factor",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--kernel",
            dest="kernel_size",
            type=int,
            default=16,
        )
        parser.add_argument(
            "--n_components", dest="n_components", type=int, default=500
        )

        parser.add_argument(
            "--prior",
            dest="prior",
            choices=["standard", "vampprior", "factorized_vampprior", "factorized_gmm"],
            default="standard",
        )


        parser.add_argument(
            "--exemplar_path", dest="exemplar_path", type=Path, default=None
        )

        return parent_parser, parser

    
if __name__ == "__main__":
    cli_main(CVAE, TrafficDataset, "image", seed=42)