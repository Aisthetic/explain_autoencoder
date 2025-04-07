from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from .responses import Autoencoder

Observation = torch.Tensor
Latent = torch.Tensor

# Custom pytorch dataset for the ILS dataset
class TrajectoriesDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, col_fid, list_features, device, scaler=None, lonlat_features=None):

        self.list_features = list_features
        self.nb_features = len(list_features)
        self.device = device
        self.scaler = scaler
        self.first_points = None
        self.last_points = None
        self.data_dir = data_dir

        # read data from datadir
        data = pd.read_parquet(data_dir)
        df = data[[col_fid] + list_features]
        df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        data_grouped = df.groupby(col_fid, sort=False).apply(lambda x: x.drop(col_fid, axis=1).to_numpy()).reset_index(name="trajectories")

        # Get number of observations per trajectory
        self.nb_obs = data_grouped["trajectories"].apply(lambda x: len(x)).min()

        # Get Flight ids form data
        self.flight_ids = data_grouped[col_fid]
        # Get flattened trajectories from data
        self.trajectories = data_grouped["trajectories"].tolist()
        # normalize data
        if scaler is not None:
            self.trajectories = self.scaler.fit_transform(np.concatenate(self.trajectories, axis=0))
            # reshape data to nb_samples , nb_features
            self.trajectories = np.array([self.trajectories[i:i+self.nb_obs] for i in range(0, len(self.trajectories), self.nb_obs)])
        # Create tensor of trajectories, convert to float and load on GPU
        self.trajectories = torch.tensor(self.trajectories, dtype=torch.float32).to(self.device)
        
        # getting lon lat features of first points of trajectories
        if lonlat_features is not None:
            # get first point of each trajectory
            self.first_points = data[[col_fid] + lonlat_features].groupby(col_fid, sort=False).head(1)
            # set index to flight id
            self.first_points = self.first_points.set_index(col_fid)
            # get last point of each trajectory
            self.last_points = data[[col_fid] + lonlat_features].groupby(col_fid, sort=False).tail(1)
            # set index to flight id
            self.last_points = self.last_points.set_index(col_fid)
            
        #print trajetories and flight ids shape and device  
        print("Trajectories shape : ", self.trajectories.shape)
        print("Flight ids shape : ", self.flight_ids.shape)
        print("Device : ", self.device)

    def df(self):
        return pd.read_parquet(self.data_dir)
            
    # Getter for the scaler
    def get_scaler(self):
        return self.scaler
        
    def __len__(self):
        return len(self.flight_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        # Get the trajectory of the flight
        trajectory = self.trajectories[idx]
        # Get the flight id
        flight_id = self.flight_ids[idx]

        return  flight_id, trajectory
    
class TrajectoriesAE(Autoencoder, nn.Module):
    def __init__(self, input_dim, encoding_dims, activation=nn.ReLU(), **kwargs):
        super().__init__(latent_dim=encoding_dims[-1], **kwargs)
        _layers = []
        encoding_dims = [input_dim] + encoding_dims
        for i in range(len(encoding_dims) - 1):
            _layers.append(nn.Linear(encoding_dims[i], encoding_dims[i+1]))
            _layers.append(activation)
        
        self.encoder = nn.Sequential(*_layers)
        
        _layers = []
        decoding_dims = encoding_dims[::-1]
        for i in range(len(decoding_dims) - 1):
            _layers.append(nn.Linear(decoding_dims[i], decoding_dims[i+1]))
            if i < len(decoding_dims) - 2 : 
                _layers.append(activation)
        
        _layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*_layers)
        
    def forward(self, x):
        # flatten the elments of the batch
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: Observation) -> Latent:
        x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def decode(self, z: Latent) -> Observation:
        return self.decoder(z)


