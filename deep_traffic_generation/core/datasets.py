# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict, Union

import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
from traffic.core import Traffic

from multiprocessing import Pool
import itertools
import os

from .protocols import TransformerProtocol

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


# fmt: on
class Infos(TypedDict):
    features: List[str]
    index: Optional[int]


class DatasetParams(TypedDict):
    features: List[str]
    file_path: Optional[Path]
    info_params: Infos
    input_dim: int
    scaler: Optional[TransformerProtocol]
    seq_len: int
    n_samples: int
    shape: str


# fmt: on
class TrafficDataset(Dataset):
    """Traffic Dataset

    Args:
        traffic: Traffic object to extract data from.
        features: features to extract from traffic.
        shape (optional): shape of datapoints when:

            - ``'image'``: tensor of shape
              :math:`(\\text{feature}, \\text{seq})`.
            - ``'linear'``: tensor of shape
              :math:`(\\text{feature} \\times \\text{seq})`.
            - ``'sequence'``: tensor of shape
              :math:`(\\text{seq}, \\text{feature})`. Defaults to
              ``'sequence'``.
        scaler (optional): scaler to apply to the data. You may want to
            consider `StandardScaler()
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.
            Defaults to None.
        info_params (Infos, optional): typed dictionnary with two keys:
            `features` (List[str]): list of features.
            `index` (int): index in the underlying trajectory DataFrame
            where to get the features.
            Defaults ``features=[]`` and ``index=None``.
    """

    _repr_indent = 4
    _available_shapes = ["linear", "sequence", "image"]

    def __init__(
        self,
        traffic: Traffic,
        features: List[str],
        shape: str = "linear",
        scaler: Optional[TransformerProtocol] = None,
        info_params: Infos = Infos(features=[], index=None),
    ) -> None:

        assert shape in self._available_shapes, (
            f"{shape} shape is not available. "
            + f"Available shapes are: {self._available_shapes}"
        )

        self.file_path: Optional[Path] = None
        self.features = features
        self.shape = shape
        self.scaler = scaler
        self.info_params = info_params

        self.data: torch.Tensor
        self.lengths: List[int]
        self.infos: List[Any]
        # self.target_transform = target_transform

        # extract labels per flight if labels column exists in traffic.data
        if "label" in traffic.data.columns:
            self.labels = [f.data[["label"]].iloc[0].values[0] for f in traffic]
            
        # extract features
        # data = extract_features(traffic, features, info_params["features"])
        data = np.stack(
            list(np.append(f.flight_id, f.data[self.features].values.ravel()) for f in traffic)
        )
        
        #keep_track of flight_id
        self.flight_ids = list(data[:,0])
        data = data[:,1:].astype(float)

        # Reshaping data to (-1, len(features)) for scaling
        if self.shape in ["sequence", "image"]:
            data = data.reshape(
                -1, len(self.features)
            )

        self.scaler = scaler
        if self.scaler is not None:
            try:
                # If scaler already fitted, only transform
                check_is_fitted(self.scaler)
                data = self.scaler.transform(data)
            except NotFittedError:
                # If not: fit and transform
                self.scaler = self.scaler.fit(data)
                data = self.scaler.transform(data)
        # Restablish shape to (nb_samples, seq_len*nb_features)
        if self.shape in ["sequence", "image"]:
            data = data.reshape(-1, len(self.features)*len(traffic[0].data))

        data = torch.FloatTensor(data)

        self.data = data
        if self.shape in ["sequence", "image"]:
            self.data = self.data.view(
                self.data.size(0), -1, len(self.features)
            )
            if self.shape == "image":
                self.data = torch.transpose(self.data, 1, 2)

        # gives info nedeed to reconstruct the trajectory
        # info_params["index"] = -1 means we need the coordinates of the last position
        self.infos = []
        # TODO: change condition (if not is_empty(self.info_params))
        if self.info_params["index"] is not None:
            self.infos = torch.Tensor(
                [
                    f.data[self.info_params["features"]]
                    .iloc[self.info_params["index"]]
                    .values.ravel()
                    for f in traffic
                ]
            )
    def inverse_transform(self, x: torch.Tensor):
        """Inverse transform the input data.

        Args:
            x (torch.Tensor): Input data to inverse transform.

        Returns:
            torch.Tensor: Inverse transformed data.
        """
        # check if is numpy array
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        nb_samples, nb_features, seq_len = x.shape
        if self.shape in ["sequence", "image"]:
            if self.shape == "image":
                x = x.transpose(0, 2, 1)
            x = x.reshape(-1, nb_features)
        x = self.scaler.inverse_transform(x)
        if self.shape in ["sequence", "image"]:
            x = x.reshape(-1, seq_len, nb_features)
            if self.shape == "image":
                x = x.transpose(0, 2, 1)
        return torch.Tensor(x)
    
    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        features: List[str],
        shape: str = "linear",
        scaler: Optional[TransformerProtocol] = None,
        info_params: Infos = Infos(features=[], index=None),
    ) -> "TrafficDataset":
        file_path = (
            file_path if isinstance(file_path, Path) else Path(file_path)
        )
        traffic = Traffic.from_file(file_path)
        dataset = cls(traffic, features, shape, scaler, info_params)
        dataset.file_path = file_path
        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, List[Any]]:
        """Get item method, returns datapoint at some index.

        Args:
            index (int): An index. Should be :math:`<len(self)`.

        Returns:
            torch.Tensor: The trajectory data shaped accordingly to self.shape.
            int: The length of the trajectory.
            list: List of informations that could be needed like, labels or
                original latitude and longitude values.
        """
        infos = []
        if self.info_params["index"] is not None:
            infos = self.infos[index]
        return self.data[index], infos
    
    def get_flight(self, flight_id: str) -> torch.Tensor:
        """Get datapoint corresponding to one flight_id in the initial traffic object

        Args:
            flight_id (str): flight_id from original traffic object

        Returns:
            torch.Tensor: datapoint associated with the given flight_id
        """
        index = self.flight_ids.index(flight_id)
        return self.data[index]

    @property
    def input_dim(self) -> int:
        """Returns the size of datapoint's features.

        .. warning::
            If the `self.shape` is ``'linear'``, the returned size will be
            :math:`\\text{feature_n} \\times \\text{sequence_len}`
            since the temporal dimension is not taken into account with this
            shape.
        """
        if self.shape in ["linear", "sequence"]:
            return self.data.shape[-1]
        elif self.shape == "image":
            return self.data.shape[1]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def seq_len(self) -> int:
        """Returns sequence length (i.e. maximum sequence length)."""
        if self.shape == "linear":
            return int(self.input_dim / len(self.features))
        elif self.shape == "sequence":
            return self.data.shape[1]
        elif self.shape == "image":
            return self.data.shape[2]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def parameters(self) -> DatasetParams:
        """Returns parameters of the TrafficDataset object in a TypedDict.

        * features (List[str])
        * file_path (Path, optional)
        * info_params (TypedDict) (see Infos for details)
        * input_dim (int)
        * scaler (Any object that matches TransformerProtocol, see TODO)
        * seq_len (int)
        * shape (str): either ``'image'``, ``'linear'`` or ```'sequence'``.
        """
        return DatasetParams(
            features=self.features,
            file_path=self.file_path,
            info_params=self.info_params,
            input_dim=self.input_dim,
            scaler=self.scaler,
            seq_len=self.seq_len,
            n_samples=self.__len__(),
            shape=self.shape,
        )

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        # if self.file_path is not None:
        #     body.append(f"File location: {self.file_path}")
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds TrafficDataset arguments to ArgumentParser.

        List of arguments:

            * ``--data_path``: The path to the traffic data file. Default to
              None.
            * ``--features``: The features to keep for training. Default to
              ``['latitude','longitude','altitude','timedelta']``.

              Usage:

              .. code-block:: console

                python main.py --features track groundspeed altitude timedelta

            * ``--info_features``: Features not passed through the model but
              useful to keep. For example, if you chose as main features:
              track, groundspeed, altitude and timedelta ; it might help to
              keep the latitude and longitude values of the first or last
              coordinates to reconstruct the trajectory. The values are picked
              up at the index specified at ``--info_index``. You can also
              get some labels.

              Usage:

              .. code-block:: console

                python main.py --info_features latitude longitude

                python main.py --info_features label

            * ``--info_index``: Index of information features. Default to None.

        Args:
            parser (ArgumentParser): ArgumentParser to update.

        Returns:
            ArgumentParser: updated ArgumentParser with TrafficDataset
            arguments.
        """
        p = parser.add_argument_group("TrafficDataset")
        p.add_argument(
            "--data_path",
            dest="data_path",
            type=Path,
            default=None,
        )
        p.add_argument(
            "--features",
            dest="features",
            nargs="+",
            default=["latitude", "longitude", "altitude", "timedelta"],
        )
        p.add_argument(
            "--info_features",
            dest="info_features",
            nargs="+",
            default=[],
        )
        p.add_argument(
            "--info_index",
            dest="info_index",
            type=int,
            default=None,
        )
        return parser

class TSDataset(Dataset):
    """Timeseries dataset adapted for https://www.timeseriesclassification.com/index.php

    Args:
        X : timeseries objects (nb_samples, nb_features, seq_len)
        y : labels (nb_samples, )
    """
    def __init__(self, X, y=None, scaler=None):
        # initialisation
        self.X = X.astype(np.float32)
        self.labels = y

        self.nb_samples = X.shape[0]
        self.nb_features = X.shape[1]
        self.seq_len = X.shape[2]

        self.input_dim = self.nb_features 
        self.shape = 'image'

        # Normalisation
        # We are going to normalize each feature across all the points of dataset 
        # We need to pass a 2D array to the scaler with shape (nb_samples*seq_len, nb_features)
        # For that, we prefer having a shape of (nb_samples, seq_len, nb_features) first
        self.X = np.swapaxes(self.X, 1, 2)
        self.scaler = scaler
        if self.scaler is not None:
            try:
                # If scaler already fitted, only transform
                check_is_fitted(self.scaler)
                self.X = self.scaler.transform(self.X.reshape(self.nb_samples*self.seq_len, self.nb_features)).reshape(self.nb_samples, self.seq_len, self.nb_features)
            except NotFittedError:
                # If not: fit and transform
                self.scaler = self.scaler.fit(self.X.reshape(self.nb_samples*self.seq_len, self.nb_features))
                self.X = self.scaler.transform(self.X.reshape(self.nb_samples*self.seq_len, self.nb_features)).reshape(self.nb_samples, self.seq_len, self.nb_features)
        self.X = np.swapaxes(self.X, 1, 2) # back to original shape

        self.X = torch.Tensor(self.X)

    def __len__(self):
        return self.nb_samples
    
    def __getitem__(self, index):
        # Check if labels are defined
        if self.labels is not None:
            return self.X[index], self.labels[index]
        return self.X[index], torch.Tensor()

    def inverse_transform(self, X):
        nb_samples, nb_features, seq_len = X.shape
        X = np.swapaxes(X, 1, 2)
        X = self.scaler.transform(X.reshape(nb_samples*seq_len, nb_features)).reshape(nb_samples, seq_len, nb_features)
        return torch.Tensor(np.swapaxes(X, 1, 2)) # back to original shape

        
    @property
    def parameters(self) -> DatasetParams:
        """Returns parameters of the TrafficDataset object in a TypedDict.

        * features (List[str])
        * file_path (Path, optional)
        * info_params (TypedDict) (see Infos for details)
        * input_dim (int)
        * scaler (Any object that matches TransformerProtocol, see TODO)
        * seq_len (int)
        * shape (str): either ``'image'``, ``'linear'`` or ```'sequence'``.
        """
        return DatasetParams(
            input_dim=self.input_dim,
            scaler=self.scaler,
            seq_len=self.seq_len,
            n_samples=self.__len__(),
            shape=self.shape,
        )

