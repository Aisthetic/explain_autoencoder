# fmt: off
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# fmt: on
class FCN(nn.Module):
    """Fully-Connected Network.

    Args:
        input_dim: size of each input sample.
        out_dim: size of each output sample.
        h_dims: list of sizes for each hidden layer.
        h_activ (optional): activation function between
            each layer. Defaults to None.
        dropout (optional): if non-zero, introduces a Dropout layer on the
            outputs of each hidden layer except the last layer, with dropout
            probability equal to :attr: `dropout`. Defaults to :math:`0.0`.
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        h_activ: Optional[nn.Module] = None,
        batch_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.n_layers = len(layer_dims) - 1
        layers = []

        for index in range(self.n_layers):
            layer = nn.Linear(
                in_features=layer_dims[index],
                out_features=layer_dims[index + 1],
            )
            layers.append(layer)

            if (index != self.n_layers - 1) and batch_norm:
                layers.append(
                    nn.BatchNorm1d(num_features=layer_dims[index + 1])
                )

            if (index != self.n_layers - 1) and h_activ is not None:
                layers.append(h_activ)

            if (index != self.n_layers - 1) and (dropout > 0):
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)

class CN(nn.Module):
    """Convolutional Network for Multivariate Time Series.

    Args:
        in_channels: Number of input features (input dimension).
        out_dim: Number of outputs.
        channels: List of output channels for each convolutional layer.
        kernel_sizes: List of kernel sizes for each convolutional layer.
        h_activ (optional): Activation function between each layer. Defaults to None.
        batch_norm: If True, adds a BatchNorm layer after each convolutional layer. Defaults to False.
        dropout: If non-zero, introduces a Dropout layer on the outputs of each convolutional layer except the last layer, with dropout probability equal to :attr: `dropout`. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        channels: List[int],
        kernel_size: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        layers = []
        current_channels = in_channels

        for out_channels in channels:
            layers.append(nn.Conv1d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same',  # Adjust padding to maintain sequence length as needed.
            ))

            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features=out_channels))

            if h_activ is not None:
                layers.append(h_activ)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_channels = out_channels

        layers.append(nn.Conv1d(
            in_channels=current_channels,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding='same',  # Adjust padding to maintain sequence length as needed.
        ))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.layers(x)
        return x
    
class RNN(nn.Module):
    """Recurrent Neural Network (LSTM based)"""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        dropout: float = 0.0,
        num_layers: int = 1,
        batch_first: bool = False,
    ) -> None:
        super().__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.n_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()

        for index in range(self.n_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=num_layers,
                batch_first=batch_first,
                # bidirectional=True,
            )
            self.layers.append(layer)

        self.dropout = dropout

    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            x, (h_n, c_n) = layer(x)

        return x, (h_n, c_n)


class TemporalBlock(nn.Module):
    """Temporal Block, a causal and dilated convolution with activation
    and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        out_activ: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation,
            )
        )

        self.out_activ = out_activ
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self) -> None:
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0), "constant", 0)
        x = self.conv(x)
        x = self.out_activ(x) if self.out_activ is not None else x
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        h_activ: Optional[nn.Module] = None,
        is_last: bool = False,
    ) -> None:
        super().__init__()

        self.is_last = is_last

        self.tmp_block1 = TemporalBlock(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            h_activ,
        )

        self.tmp_block2 = TemporalBlock(
            out_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            h_activ if not is_last else None,  # deactivate last activation
        )

        # Optional convolution for matching in_channels and out_channels sizes
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        # self.out_activ = h_activ
        self.init_weights()

    def init_weights(self) -> None:
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.tmp_block1(x)
        y = self.tmp_block2(y)
        r = x if self.downsample is None else self.downsample(x)
        return y + r


class TCN(nn.Module):
    """Temporal Convolutional Network.

    Handle data shaped :math:`(N, C, L)` with :math:`N` the batch size,
    :math:`C` the number of channels and :math:`L` the length of signal
    sequence.

    Source:
        Adaptation from the code source of the paper `An Empirical Evaluation
        of Generic Convolutional and Recurrent Networks for Sequence Modeling
        <https://arxiv.org/abs/1803.01271>`_ by Shaojie Bai, J. Zico Kolter and
        Vladlen Koltun.

    Args:
        input_dim (int): number of channels in the input sequence.
        out_dim (int): number of channels produced by the network.
        h_dims (List[int]): number of channels outputed by the hidden layers.
        kernel_size (int): size of the convolving kernel.
        dilation_base (int): size of the dilation base to compute the dilation
            of a specific layer.
        activation (torch.nn.Module): activation function to use within
            temporal blocks.
            batch_norm (bool): whether to use batch normalization or not between TCN blocks.
        dropout (float, optional): probability of an element to be zeroed.
            Default :math:`0.2`.

    .. note::
        Keep the kernel size at least as big as the dilation base to avoid any
        holes in your receptive field i.e.

        .. math::
            \\text{kernel_size} \\geq \\text{dilatation_base}
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        kernel_size: int,
        dilation_base: int,
        batch_norm: Optional[bool] = False,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        layer_dims = [input_dim] + h_dims + [out_dim]
        self.n_layers = len(layer_dims) - 1
        layers = []

        for index in range(self.n_layers):
            dilation = dilation_base ** index
            in_channels = layer_dims[index]
            out_channels = layer_dims[index + 1]
            is_last = index == (self.n_layers - 1)
            layer = ResidualBlock(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                dropout,
                h_activ,
                is_last,
            )
            layers.append(layer)
            if batch_norm and not is_last:
                layers.append(nn.BatchNorm1d(out_channels))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)