# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
import math
from numbers import Number
from typing import Dict, Tuple, Union
from xmlrpc.client import boolean

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.distributions.distribution import Distribution
from torch.distributions import Independent, Normal, MixtureSameFamily
from torch.nn import functional as F
from cartes.crs import EuroPP
from torch.distributions.categorical import Categorical
from deep_traffic_generation.core.datasets import DatasetParams

import numpy as np

from .builders import (
    CollectionBuilder, IdentifierBuilder, LatLonBuilder, TimestampBuilder
)
from .utils import plot_traffic, traffic_from_data


# fmt: on
class LSR(LightningModule):
    """Abstract for Latent Space Regularization Networks.

    Args:
        input_dim (int): size of each input sample.
        out_dim (int): size pf each output sample.
        fix_prior (bool, optional): Whether to optimize the prior distribution.
            Defaults to ``True``.
    """

    def __init__(self, input_dim: int, out_dim: int, fix_prior: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.fix_prior = fix_prior

    def forward(self, hidden: torch.Tensor) -> Distribution:
        """Defines the computation performed at every call.

        Returns a Distribution object according to a tensor. Ideally the
        Distribution object implements a rsample() method.

        .. note::
            Should be overriden by all subclasses.
        """
        raise NotImplementedError()

    def dist_params(self, p: Distribution) -> Tuple:
        """Returns a tuple of tensors corresponding to the parameters of
        a given distribution.

        .. note::
            Should be overriden by all subclasses.
        """
        raise NotImplementedError()

    def get_posterior(self, dist_params: Tuple) -> Distribution:
        """Returns a Distribution object according to a tuple of parameters.
        Inverse method of dist_params().

        Args:
            dist_params (Tuple): tuple of tensors corresponding to distribution
                parameters.

        .. note::
            Should be overriden by all subclasses.
        """
        raise NotImplementedError()

    def get_prior(self, batch_size: int) -> Distribution:
        """Returns the prior distribution we want the posterior distribution
        to fit.

        .. note::
            Should be overriden by all subclasses.
        """
        raise NotImplementedError()


class Abstract(LightningModule):
    """Abstract class for deep models."""

    _required_hparams = [
        "lr",
        "lr_step_size",
        "lr_gamma",
        "dropout",
    ]

    def __init__(
        self, dataset_params: DatasetParams, config: Union[Dict, Namespace]
    ) -> None:
        super().__init__()

        self._check_hparams(config)
        self.save_hyperparameters(config)

        self.dataset_params = dataset_params

        # self.criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    @classmethod
    def network_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def add_model_specific_args(
        cls,
        parent_parser: ArgumentParser,
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        parser = parent_parser.add_argument_group(f"{cls.network_name()}")
        parser.add_argument(
            "--name",
            dest="network_name",
            default=f"{cls.network_name()}",
            type=str,
            help="network name",
        )
        parser.add_argument(
            "--lr",
            dest="lr",
            default=1e-3,
            type=float,
            help="learning rate",
        )
        parser.add_argument(
            "--lrstep",
            dest="lr_step_size",
            default=100,
            type=int,
            help="period of learning rate decay (in epochs)",
        )
        parser.add_argument(
            "--lrgamma",
            dest="lr_gamma",
            default=1.0,
            type=float,
            help="multiplicative factor of learning rate decay",
        )
        parser.add_argument(
            "--dropout", dest="dropout", type=float, default=0.0
        )

        return parent_parser, parser

    def get_builder(self, nb_samples: int, length: int) -> CollectionBuilder:
        builder = CollectionBuilder(
            [
                IdentifierBuilder(nb_samples, length),
                TimestampBuilder(),
            ]
        )
        if "track_unwrapped" in self.dataset_params["features"]:
            if self.dataset_params["info_params"]["index"] == 0:
                builder.append(LatLonBuilder(build_from="azgs"))
            elif self.dataset_params["info_params"]["index"] == -1:
                builder.append(LatLonBuilder(build_from="azgs_r"))
        elif "track" in self.dataset_params["features"]:
            if self.dataset_params["info_params"]["index"] == 0:
                builder.append(LatLonBuilder(build_from="azgs"))
            elif self.dataset_params["info_params"]["index"] == -1:
                builder.append(LatLonBuilder(build_from="azgs_r"))
        elif "x" in self.dataset_params["features"]:
            builder.append(LatLonBuilder(build_from="xy", projection=EuroPP()))

        return builder

    def _check_hparams(self, hparams: Union[Dict, Namespace]):
        for hparam in self._required_hparams:
            if isinstance(hparams, Namespace):
                if hparam not in vars(hparams).keys():
                    raise AttributeError(
                        f"Can't set up network, {hparam} is missing."
                    )
            elif isinstance(hparams, dict):
                if hparam not in hparams.keys():
                    raise AttributeError(
                        f"Can't set up network, {hparam} is missing."
                    )
            else:
                raise TypeError(f"Invalid type for hparams: {type(hparams)}.")


class AE(Abstract):
    """Abstract class for Autoencoders.

    Usage Example:
        .. code:: python

            import torch.nn as nn
            from deep_traffic_generation.core import AE

            class YourAE(AE):
                def __init__(self, dataset_params, config):
                    super().__init__(dataset_params, config)

                    # Define encoder
                    self.encoder = nn.Linear(64, 16)

                    # Define decoder
                    self.decoder = nn.Linear(16, 64)
    """

    _required_hparams = Abstract._required_hparams + [
        "encoding_dim",
        "h_dims",
    ]

    def __init__(
        self, dataset_params: DatasetParams, config: Union[Dict, Namespace]
    ) -> None:
        super().__init__(dataset_params, config)

        self.encoder: nn.Module
        self.decoder: nn.Module
        self.out_activ: nn.Module

    def configure_optimizers(self) -> dict:
        """Optimizers."""
        # optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_step_size,
            gamma=self.hparams.lr_gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams
        )
        self.logger.log_metrics(
             {"hp/valid_loss": 1, "hp/test_loss": 1}
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.out_activ(self.decoder(z))
        return z, x_hat

    def training_step(self, batch, batch_idx):
        """Training step.

        The validation loss is the Mean Square Error
        :math:`\\mathcal{L}_{MSE}(x_{i}, \\hat{x_{i}})`.
        """
        x, _ = batch
        _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        The validation loss is the Mean Square Error
        :math:`\\mathcal{L}_{MSE}(x_{i}, \\hat{x_{i}})`.
        """
        x, _ = batch
        _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", np.log(loss.cpu().item()), sync_dist=True)

    def test_step(self, batch, batch_idx):
        """Test step.

        The test loss is the Mean Square Error
        :math:`\\mathcal{L}_{MSE}(x_{i}, \\hat{x_{i}})`.
        """
        x, info = batch
        _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", np.log(loss.cpu().item()), sync_dist=True)
        return x, x_hat, info

    def test_epoch_end(self, outputs) -> None:
        """FIXME: too messy."""
        idx = 0
        original = outputs[0][0][idx].unsqueeze(0).cpu()
        reconstructed = outputs[0][1][idx].unsqueeze(0).cpu()
        data = torch.cat((original, reconstructed), dim=0)
        data = data.reshape((data.shape[0], -1))
        # unscale the data
        if self.dataset_params["scaler"] is not None:
            data = self.dataset_params["scaler"].inverse_transform(data)

        if isinstance(data, torch.Tensor):
            data = data.numpy()
        # add info if needed (init_features)
        if len(self.dataset_params["info_params"]["features"]) > 0:
            info = outputs[0][2][idx].unsqueeze(0).cpu().numpy()
            info = np.repeat(info, data.shape[0], axis=0)
            data = np.concatenate((info, data), axis=1)
        # get builder
        builder = self.get_builder(
            nb_samples=2, length=self.dataset_params["seq_len"]
        )
        features = [
            "track" if "track" in f else f for f in self.hparams.features
        ]
        # build traffic
        traffic = traffic_from_data(
            data,
            features,
            self.dataset_params["info_params"]["features"],
            builder=builder,
        )
        # generate plot then send it to logger
        self.logger.experiment.add_figure(
            "original vs reconstructed", plot_traffic(traffic)
        )

    @classmethod
    def add_model_specific_args(
        cls,
        parent_parser: ArgumentParser,
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds AE arguments to ArgumentParser.

        List of arguments:

            * ``--encoding_dim``: Latent space size. Default to :math:`32`.
            * ``--h_dims``: List of dimensions for hidden layers. Default to
              ``[]``.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--encoding_dim",
            dest="encoding_dim",
            type=int,
            default=32,
        )
        parser.add_argument(
            "--h_dims",
            dest="h_dims",
            nargs="+",
            type=int,
            default=[],
        )

        return parent_parser, parser


class VAE(AE):
    
    """Abstract class for Variational Autoencoder. Adaptation of the VAE
    presented by William Falcon in `Variational Autoencoder Demystified With
    PyTorch Implementation
    <https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed>`_.

    Usage Example:
        .. code:: python

            import torch.nn as nn
            from deep_traffic_generation.core import NormalLSR, VAE

            class YourVAE(VAE):
                def __init__(self, dataset_params, config):
                    super().__init__(dataset_parms, config)

                    # Define encoder
                    self.encoder = nn.Linear(64, 32)

                    # Example of latent space regularization
                    self.lsr = NormalLSR(
                        input_dim=32,
                        out_dim=16
                    )

                    # Define decoder
                    self.decoder = nn.Sequential(
                        nn.Linear(16, 32),
                        nn.ReLU(),
                        nn.Linear(32, 64)
                    )
    """

    _required_hparams = AE._required_hparams + [
        "kld_coef",
        "llv_coef",
        "scale",
        # "fix_prior",
    ]

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        # Regularization parameter for pseudo inputs
        self.pseudo_gamma = 0.1

        # Auto balancing between kld and llv with the decoder scale
        # Diagnosing and Enhancing VAE Models
        self.scale = nn.Parameter(
            torch.Tensor([self.hparams.scale]), requires_grad=True
        )

        # Latent Space Regularization
        self.lsr: LSR

    def forward(self, x) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        # encode x to get the location and log variance parameters
        h = self.encoder(x)
        # When batched, q is a collection of normal posterior
        q = self.lsr(h)
        z = q.rsample()
        # decode z
        x_hat = self.out_activ(self.decoder(z))
        return self.lsr.dist_params(q), z, x_hat

    def training_step(self, batch, batch_idx):
        """Training step.

        Computes the gaussian likelihood and the Kullback-Leibler divergence
        to get the ELBO loss function.

        .. math::

            \\mathcal{L}_{ELBO} = \\alpha \\times KL(q(z|x) || p(z))
            - \\beta \\times \\sum_{i=0}^{N} log(p(x_{i}|z_{i}))
        """
        x, _ = batch
        dist_params, z, x_hat = self.forward(x)

        # std of decoder distribution (init at 1)
        # self.scale = nn.Parameter(
        #     torch.Tensor([torch.sqrt(F.mse_loss(x, x_hat))]),
        #     requires_grad=False,
        # )
        # gamma = self.scale

        # Regular VAE LOSS
        # log likelihood loss (reconstruction loss)
        llv_loss = -self.gaussian_likelihood(x, x_hat)
        llv_coef = self.hparams.llv_coef
        # kullback-leibler divergence (regularization loss)
        q_zx = self.lsr.get_posterior(dist_params)
        p_z = self.lsr.get_prior()
        kld_loss = self.kl_divergence(z, q_zx, p_z)
        kld_coef = self.hparams.kld_coef

        # elbo with beta hyperparameter:
        #   Higher values enforce orthogonality between latent representation.
        elbo = kld_coef * kld_loss + llv_coef * llv_loss
        elbo = elbo.mean()

        # Regularization to make the pseudo-inputs close to their reconstruction
        if self.hparams.reg_pseudo:
            # Calculate pseudo-inputs for regularization term
            pseudo_X = self.lsr.pseudo_inputs_NN(self.lsr.idle_input)
            pseudo_X = pseudo_X.view(
                (pseudo_X.shape[0], x.shape[1], x.shape[2])
            )
            pseudo_dist_params, pseudo_z, pseudo_x_hat = self.forward(pseudo_X)

            # Regularization term for pseudo_inputs
            # log likelihood loss (reconstruction loss)
            pseudo_llv_loss = -self.gaussian_likelihood(pseudo_X, pseudo_x_hat)
            # kullback-leibler divergence (regularization loss)
            pseudo_q_zx = self.lsr.get_posterior(pseudo_dist_params)
            pseudo_kld_loss = self.kl_divergence(pseudo_z, pseudo_q_zx, p_z)

            pseudo_elbo = (
                kld_coef * pseudo_kld_loss + llv_coef * pseudo_llv_loss
            )
            pseudo_elbo = (x.shape[0] / pseudo_X.shape[0]) * pseudo_elbo.mean()
            elbo = elbo + self.pseudo_gamma * pseudo_elbo

        # ELBO loss from Diagnosing and Enhancing VAE Models
        # Values are very close, but we can have access to the gamma parameter
        # kl = (
        #     torch.sum(self.kl_loss(dist_params[1], dist_params[2])) / batch_size
        # )
        # gen = torch.sum(self.gen_loss(x, x_hat, gamma)) / batch_size
        # elbo = kl + gen

        self.log_dict(
            {
                "train_loss": F.mse_loss(x_hat, x),
                "train_elbo": elbo,
                "kl_loss": kld_loss.mean(),
                "recon_loss": llv_loss.mean(),
            }
        )
        return elbo

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", np.log(loss.cpu().item()), sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, info = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", np.log(loss.cpu().item()), sync_dist=True)
        return x, x_hat, info

    def gaussian_likelihood(self, x: torch.Tensor, x_hat: torch.Tensor):
        """Computes the gaussian likelihood.

        Args:
            x (torch.Tensor): input data
            x_hat (torch.Tensor): mean decoded from :math:`z`.

        .. math::

            \\sum_{i=0}^{N} log(p(x_{i}|z_{i}))
            \\text{ with } p(.|z_{i})
            \\sim \\mathcal{N}(\\hat{x_{i}},\\,\\sigma^{2})

        .. note::
            The scale :math:`\\sigma` can be defined in config and will be
            accessible with ``self.scale``.
        """
        mean = x_hat
        dist = torch.distributions.Normal(mean, self.scale)
        # measure prob of seeing trajectory under p(x|z)
        log_pxz = dist.log_prob(x)
        dims = [i for i in range(1, len(x.size()))]
        return log_pxz.sum(dim=dims)

    def kl_divergence(
        self, z: torch.Tensor, p: Distribution, q: Distribution
    ) -> torch.Tensor:
        """Computes Kullback-Leibler divergence :math:`KL(p || q)` between two
        distributions, using Monte Carlo Sampling.
        Evaluate every z of the batch in its corresponding posterior (1st z with 1st post, etc..)
        and every z in the prior

        Args:
            z (torch.Tensor): A sample from p (the posterior).
            p (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the posetrior)
            q (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the prior)

        Returns:
            torch.Tensor: A batch of KL divergences of shape `z.size(0)`.

        .. note::
            Make sure that the `log_prob()` method of both Distribution
            objects returns a 1D-tensor with the size of `z` batch size.
        """
        log_p = p.log_prob(z)
        log_q = q.log_prob(z)
        return log_p - log_q

    # The 2 terms here are part of the other formulation of the loss
    # present in Diagnosing and Enhancing VAE Models
    def gen_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, gamma: torch.Tensor
    ):
        """Computes generation loss in TwoStages VAE Model
        Args :
            x : input data
            x_hat : reconstructed data
            gamma : decoder std (scalar as every distribution in the decoder has the same std)

        To use it within the learning : take the sum and divide by the batch size
        """
        HALF_LOG_TWO_PI = 0.91893

        loggamma = torch.log(gamma)
        return (
            torch.square((x - x_hat) / gamma) / 2.0 + loggamma + HALF_LOG_TWO_PI
        )

    def kl_loss(self, mu: torch.Tensor, std: torch.Tensor):
        """Computes close form of KL for gaussian distributions
        Args :
            mu : encoder means
            std : encoder stds

        To use it within the learning : take the sum and divide by the batch size
        """
        logstd = torch.log(std)
        return (torch.square(mu) + torch.square(std) - 2 * logstd - 1) / 2.0

    def reconstruct(self, x):
        """Reconstruct a batch of data.

        Args:
            x (torch.Tensor): Batch of data to reconstruct.

        Returns:
            torch.Tensor: Reconstructed batch of data.
        """
        _, _, x_hat = self.forward(x)
        return x_hat
    
    def encode(self, x):
        # encode x to get the location and log variance parameters
        h = self.encoder(x)
        # When batched, q is a collection of normal posterior
        z = self.lsr(h)
        return z
        
    def decode(self, z):
        # check if z is not a tensor
        if not isinstance(z, torch.Tensor):
            z = z.rsample()
        # decode z
        return self.out_activ(self.decoder(z))
    
    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds VAE arguments to ArgumentParser.

        List of arguments:

            * ``--llv_coef``: Coefficient for the gaussian log likelihood
              (reconstruction loss): :math:`\\beta`.
            * ``--kld_coef``: Coefficient for the Kullback-Leibler divergence
              (regularization loss): :math:`\\alpha`.
            * ``--scale``: Define the scale :math:`\\sigma` of the Normal law
              used to sample the reconstruction.
            * `fix-prior`: Whether the prior is learnable or not. Default to
              ``False``.

                * ``--fix-prior``: is not learnable;
                * ``--no-fix-prior``: is learnable.

        .. note::
            It adds also the argument of the inherited class `AE`.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--llv_coef",
            dest="llv_coef",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--kld_coef", dest="kld_coef", type=float, default=1.0
        )

        parser.add_argument(
            "--reg_pseudo", dest="reg_pseudo", type=boolean, default=False
        )

        parser.add_argument("--scale", dest="scale", type=float, default=1.0)
        parser.add_argument(
            "--fix-prior", dest="fix_prior", action="store_true"
        )
        parser.add_argument(
            "--no-fix-prior", dest="fix_prior", action="store_false"
        )
        parser.set_defaults(fix_prior=True)

        return parent_parser, parser
    
class VAE_disent(AE):
    """Abstract class for Variational Autoencoder. Adaptation of the VAE 
    to genreate pairs of trajectories in a disentangled manner.
    inspired by: https://arxiv.org/abs/1802.04942
    
    Works only for a factorised gaussian posterior and a factorised prior
    """

    _required_hparams = AE._required_hparams + [
        "llv_coef",
        "tc_coef",
        "kld_coef",
        # "post_coef",
        # "prior_coef",
        "scale"
    ]

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.encoder: nn.Module
        self.decoder: nn.Module

        # Auto balancing between kld and llv with the decoder scale
        # Diagnosing and Enhancing VAE Models
        # Scales for the decoders
        self.scale_traj = nn.Parameter(
            torch.Tensor([self.hparams.scale]), requires_grad=True
        )

        # Latent Space Regularization
        self.lsr: LSR

    def forward(self, x) -> Tuple[Tuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
    
        # q is a collection of normal posterior for each traj of the batch
        q = self.lsr(h)
        z = q.rsample()

        # decode z into reconstructed pair
        x_hat = self.decoder(z)
        x_hat = self.out_activ(x_hat)
        return self.lsr.dist_params(q), z, x_hat


    def training_step(self, batch, batch_idx):
        """Training step.

        Computes the ELBO according to the decomposition in https://arxiv.org/abs/1802.04942

        """
        
        x, _ = batch
        dist_params, z, x_hat = self.forward(x)
        
        # prior
        pz = self.lsr.get_prior()
        if len(pz.event_shape) > 1:
            logpz = pz.log_prob(z.unsqueeze(2))
        else:
            logpz = pz.log_prob(z)

        # log likelihood loss for pairs of trajectories
        llv_loss = -self.gaussian_likelihood(x, x_hat, self.scale_traj)
        llv_coef = self.hparams.llv_coef
        
        # Calculation of log q(z_i|x_i) (the x_i used to generate z_i)
        q_zx = self.lsr.get_posterior(dist_params)
        logq_zx = q_zx.log_prob(z) #size (n_batch,)
        
        # kullback-leibler divergence  in the regular elbo
        q_zx = self.lsr.get_posterior(dist_params)
        kld_loss = self.kl_divergence(z, q_zx, pz)
        
        # Calculation of log q(z|x_i) for i = 1, ..., n_batch with z fixed
        # logq_zx corresponds to the diagonal of logq_zx_extended
        logqz_extended = q_zx.log_prob(z.view(-1, 1, z.shape[1])) # size (n_batch, n_batch)
        
        # Estimation of q(z) and product of the marginals of q(z) with minibatch weighted sampling: logq(z) ~= - log(MN) + logsumexp_m(q(z|x_m))
        marginals = Normal(dist_params[0], dist_params[1]).log_prob(z.view(-1, 1, z.shape[1])) # size (n_batch, n_batch, latent_dim)
        logqz_prodmarginals = (self.logsumexp(marginals, dim = 1) - math.log(z.shape[0] * self.dataset_params["n_samples"])).sum(1) 
        logqz = self.logsumexp(logqz_extended, dim = 1) - math.log(z.shape[0] * self.dataset_params["n_samples"])
        
        # # We introduce a regularization on the prior to force it to be factorized (case when the prior is VampPrior)
        # # The marginals of a Gaussian mixture with factorized components are the mixture of the marginals
        if self.hparams.post_coef > 0 or self.hparams.prior_coef > 0:
            pz_cat_probs = Categorical(probs = pz.mixture_distribution.probs.repeat(z.shape[1],1))
            pz_mu = pz.component_distribution.base_dist.loc.T.unsqueeze(2)
            pz_sigma = pz.component_distribution.base_dist.scale.T.unsqueeze(2)
            pz_marginals_comp = Independent(Normal(pz_mu, pz_sigma), 1)
            pz_marginals = MixtureSameFamily(pz_cat_probs, pz_marginals_comp)
            logpz_prodmarginals = self.logsumexp(pz_marginals.log_prob(z.unsqueeze(2)), dim = 1)
        
        # Mutual information loss
        mi_loss = (logq_zx - logqz)
        
        # Total Correlation loss
        tc_loss = (logqz - logqz_prodmarginals)
        tc_coef = self.hparams.tc_coef
        
        # Element-wise KL divergence loss
        kld_ew_loss = (logqz_prodmarginals - logpz) 
        # #If pz is not factorized: problem. We would prefer to use logpz_prodmarginals if we keep the "dimension-wise kl" characteristic
        # kld_ew_loss = (logqz_prodmarginals - logpz_prodmarginals)
        kld_coef = self.hparams.kld_coef
        
        # elbo
        elbo = kld_loss + llv_coef * llv_loss
        elbo = elbo.detach().mean()
        modified_elbo = llv_coef * llv_loss  + mi_loss + tc_coef * tc_loss + kld_coef * kld_ew_loss 
        
        # Aggregated posterior regularization: https://arxiv.org/abs/1812.02833
        if self.hparams.post_coef > 0:
            post_loss = (logqz - logpz)
            post_coef = self.hparams.post_coef
            modified_elbo = modified_elbo + post_coef * post_loss
        else:
            post_loss = torch.zeros_like(logqz).detach()
            post_coef = 0
        
        # Factorized prior regularization (force VampPrior to be factorized)
        if self.hparams.prior_coef > 0:
            prior_loss = -(logpz - logpz_prodmarginals)
            prior_coef = self.hparams.prior_coef 
            modified_elbo = modified_elbo + prior_coef * prior_loss
        else:
            prior_loss = torch.zeros_like(logpz).detach()
            prior_coef = 0

        modified_elbo = modified_elbo.mean()

        self.log_dict(
            {
                "train_loss": modified_elbo,
                "elbo": elbo,
                "recon_loss": llv_loss.mean(),
                "mutual_information": mi_loss.mean(),
                "total_correlation": tc_loss.mean(),
                "element_wise_kl_divergence": kld_ew_loss.mean(),
                "error_est_kld": torch.abs(kld_loss.mean() - (mi_loss + tc_loss + kld_ew_loss).mean()),
                "posterior_reg": post_loss.mean(),
                "prior_reg": prior_loss.mean(),
            }
        )
        return modified_elbo

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", np.log(loss.cpu().item()), sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x) 
        self.log("hp/test_loss", np.log(loss.cpu().item()), sync_dist=True)
        return x, x_hat
    
    def logsumexp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0),
                                        dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            if isinstance(sum_exp, Number):
                return m + math.log(sum_exp)
            else:
                return m + torch.log(sum_exp)

    def gaussian_likelihood(self, x: torch.Tensor, x_hat: torch.Tensor, scale):
        """Computes the gaussian likelihood.

        Args:
            x (torch.Tensor): input data
            x_hat (torch.Tensor): mean decoded from :math:`z`.

        .. math::

            \\sum_{i=0}^{N} log(p(x_{i}|z_{i}))
            \\text{ with } p(.|z_{i})
            \\sim \\mathcal{N}(\\hat{x_{i}},\\,\\sigma^{2})

        .. note::
            The scale :math:`\\sigma` can be defined in config and will be
            accessible with ``self.scale``.
        """
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing trajectory under p(x|z)
        log_pxz = dist.log_prob(x)
        dims = [i for i in range(1, len(x.size()))]
        return log_pxz.sum(dim=dims)

    def kl_divergence(
        self, z: torch.Tensor, p: Distribution, q: Distribution
    ) -> torch.Tensor:
        """Computes Kullback-Leibler divergence :math:`KL(p || q)` between two
        distributions, using Monte Carlo Sampling.
        Evaluate every z of the batch in its corresponding posterior (1st z with 1st post, etc..)
        and every z in the prior

        Args:
            z (torch.Tensor): A sample from p (the posterior).
            p (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the posetrior)
            q (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the prior)

        Returns:
            torch.Tensor: A batch of KL divergences of shape `z.size(0)`.

        .. note::
            Make sure that the `log_prob()` method of both Distribution
            objects returns a 1D-tensor with the size of `z` batch size.
        """
        if len(p.event_shape) > 1:
            log_p = p.log_prob(z.unsqueeze(2))
        else:
            log_p = p.log_prob(z)
        if len(q.event_shape) > 1:
            log_q = q.log_prob(z.unsqueeze(2))
        else:
            log_q = q.log_prob(z)
        return log_p - log_q
        
    def reconstruct(self, x):
        """Reconstruct a batch of data.

        Args:
            x (torch.Tensor): Batch of data to reconstruct.

        Returns:
            torch.Tensor: Reconstructed batch of data.
        """
        _, _, x_hat = self.forward(x)
        return x_hat
    
    def encode(self, x):
        # encode x to get the location and log variance parameters
        h = self.encoder(x)
        # When batched, q is a collection of normal posterior
        z = self.lsr(h)
        return z
        
    def decode(self, z):
        # check if z is not a tensor
        if not isinstance(z, torch.Tensor):
            z = z.rsample()
        # decode z
        return self.out_activ(self.decoder(z))

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds VAE arguments to ArgumentParser.

        List of arguments:

            * ``--llv_coef``: Coefficient for the gaussian log likelihood
            * ``--kld_coef``: Coefficient for the dimension-wise Kullback-Leibler divergence
            * ``--tc_coef``: Coefficient for the total correlation
            * ``--post_coef``: Coefficient for the divergence between the aggregated posterior and the prior
            * ``--prior_coef``: Coefficient for the total correlation of the prior
            * ``--scale``: Define the scale :math:`\\sigma` of the Normal law
            used to sample the reconstruction.

        .. note::
            It adds also the argument of the inherited class `AE`.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--llv_coef",
            dest="llv_coef",
            type=float,
            default=1.0,
        )
        
        parser.add_argument(
            "--kld_coef", dest="kld_coef", type=float, default=1.0
        )
        
        parser.add_argument(
            "--tc_coef", dest="tc_coef", type=float, default=1.0
        )
        
        parser.add_argument(
            "--post_coef", dest="post_coef", type=float, default=0.0
        )
        
        parser.add_argument(
            "--prior_coef", dest="prior_coef", type=float, default=0.0
        )

        parser.add_argument("--scale", dest="scale", type=float, default=1.0)

        return parent_parser, parser