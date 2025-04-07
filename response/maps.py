from turtle import color
from typing import Union, Optional, TypeVar, Generic, Dict, List
from matplotlib import axes

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import distributions as distrib
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from .responses import Autoencoder
from .utils import build_trajectories, plot_imgs, plot_trajectories

from matplotlib.colors import Normalize
import matplotlib.image as mpimg
from matplotlib import cm



def push_forward(autoencoder: Autoencoder, data: Union[Dataset, torch.Tensor], fn_name: str,
                 num_samples: Optional[int] = None, batch_size: int = 1024,
                 device: Optional[str] = None, pbar: Optional[tqdm] = None) -> torch.Tensor:
    if num_samples is None:
        num_samples = len(data)
    else:
        num_samples = min(num_samples, len(data))
    batches = iter(torch.arange(len(data)).split(batch_size))

    outs = []
    N = 0

    fn = getattr(autoencoder, fn_name)

    if pbar is not None:
        pbar = pbar(total=num_samples)
    while N < num_samples:
        batch = next(batches)
        X = data[batch]
        if isinstance(X, tuple):
            X = X[0]
        out = fn(X.to(device))
        if isinstance(out, distrib.Distribution):
            out = out.mean
        outs.append(out.cpu())
        N += len(batch)
        if pbar is not None:
            pbar.update(len(batch))

    if pbar is not None:
        pbar.close()
    return torch.cat(outs)


def collect_posterior_means(autoencoder: Autoencoder, data: Union[Dataset, torch.Tensor], **kwargs):
    return push_forward(autoencoder, data, 'encode', **kwargs)


def collect_reconstructions(autoencoder: Autoencoder, data: Union[Dataset, torch.Tensor], **kwargs):
    return push_forward(autoencoder, data, 'reconstruct', **kwargs)


def collect_responses(autoencoder: Autoencoder, data: Union[Dataset, torch.Tensor], **kwargs):
    return push_forward(autoencoder, data, 'response', **kwargs)



def generate_2d_latent_map(xidx, yidx, base, n=64, h=None, w=None, extent=None):
    xmin, xmax, ymin, ymax = extent
    if h is None:
        h = n
    if w is None:
        w = n

    cx, cy = torch.meshgrid(torch.linspace(xmin, xmax, h), torch.linspace(ymin, ymax, w))
    cx = cx.reshape(-1)
    cy = cy.reshape(-1)

    vecs = base.cpu().view(1, -1).expand(len(cx), -1).contiguous()
    vecs[:, xidx] = cx
    vecs[:, yidx] = cy
    return vecs.view(h, w, -1)



def collect_response_map(autoencoder: Autoencoder, zmap: torch.Tensor, batch_size: int = 128,
                 device: Optional[str] = None, pbar: Optional[tqdm] = None) -> torch.Tensor:
    '''
    Compute the response map for a given set of inputs `X`.
    :param autoencoder: used for computing the response (decode + encode)
    :param zmap: input latent sample map (H, W, latent_dim)
    :param batch_size: max batch size
    :param device: optional device
    :param pbar: optional progress bar
    :return: response map (H, W, latent_dim)
    '''

    H, W, _ = zmap.shape
    responses = collect_responses(autoencoder, zmap.view(H*W, -1), batch_size=batch_size, device=device, pbar=pbar)
    return responses.view(H, W, -1)


def response_map_2d(autoencoder: Autoencoder, xidx, yidx, base=None, n=64, batch_size: int = 128,
                    device: Optional[str] = None, pbar: Optional[tqdm] = None, **kwargs):

    if base is None:
        base = autoencoder.sample_prior(1)

    zmap = generate_2d_latent_map(xidx, yidx, base, n=n, **kwargs)
    rmap = collect_response_map(autoencoder, zmap, batch_size, device, pbar)
    umap = rmap.sub(zmap)[..., [xidx, yidx]]
    return umap


def compute_mean_curvature_2d(umap):
    nmap = umap.div(umap.norm(dim=-1, keepdim=True))
    cmap = - 0.5 * compute_divergence_2d(nmap)
    return cmap


def compute_divergence_2d(deltas: torch.Tensor) -> torch.Tensor:
    '''
    Numerically estimates the divergence by finite differencing of a given 2D grid.

    :param deltas: grid of 2D vectors (H, W, 2)
    :return: divergences (H, W)
    '''
    deltas = deltas.cpu().detach().numpy()
    divx, divy = np.gradient(deltas, axis=[0,1])
    divM = divx[...,0] + divy[...,1]
    return torch.as_tensor(divM)

def curvature(x, y):
    '''
    Compute the mean curvature of a 2D curve
    
    :param x: x coordinates of the curve
    :param y: y coordinates of the curve
    :return: mean curvature
    '''
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num = dx * ddy - dy * ddx
    den = np.power(dx ** 2 + dy ** 2, 1.5)
    den[den == 0] = 1e-5  # To avoid division by zero
    curv = num / den
    mean_curv = np.mean(np.abs(curv))
    return mean_curv

def curvature_map_2d(df, n, seq_len):
    H, W = n,n 
    curvature_values = np.zeros((30, 30, 2))
    # get longitude and latitude from df
    trajectories = df[['longitude', 'latitude']]
    # reshape to (H*W, seq_len, 2)
    trajectories = trajectories.values.reshape(-1, seq_len, 2)
    # compute curvature for each trajectory
    for i in range(30):
        for j in range(30):
            idx = i * 30 + j
            traj = trajectories[idx]
            mean_curv = curvature(traj[:,0].values, traj[:,1].values)
            curvature_values[i, j, :] = [mean_curv, mean_curv]

    return curvature_values


def plot_map(grid, rescale=True, fgax=None, aspect=None, cmap='viridis', colorbar=False, colorbar_fontsize=10, **kwargs):
    if fgax is not None:
        fg, ax = fgax
        plt.sca(ax)
    else:
        fg, ax = plt.subplots()

    if aspect is None:
        aspect = grid.size(1) / grid.size(0)

    axvals = np.linspace(grid.min().item(), grid.max().item(), 9)
    if rescale:
        axvals = np.concatenate([np.linspace(grid.min().item(), 0, 5), np.linspace(0, grid.max().item(), 5)[1:]])
        grid = rescale_map_for_viz(grid.clone())  

    
    out = plt.imshow(grid.detach().cpu().t().numpy(), cmap=cmap, aspect=aspect, **kwargs)
    
    if colorbar:
        cbar = plt.colorbar(ax=ax)
        axlbls = [f'{v.item():.2f}' for v in axvals]
        cbar.set_ticks(np.linspace(-1, 1, 9))
        cbar.set_ticklabels(axlbls)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)  # Change font size here

    return out


def plot_map_latex(grid, xlabel='', ylabel='', title='', rescale=True, fgax=None, 
             aspect=None, cmap='viridis', colorbar=False, colorbar_fontsize=11, 
             fontsize=11, linewidth=1, save_path='', **kwargs):
    """
    Plot a grid map with customization options suitable for IEEE papers.

    Parameters:
        grid : array-like
            Data to plot.
        xlabel, ylabel, title : str
            Labels and title for the plot.
        rescale : bool
            If True, rescale the grid for visualization.
        fgax : tuple
            (figure, axis) objects (if None, new objects are created).
        aspect : float or str
            Aspect ratio of the plot.
        cmap : str
            Colormap to use for the plot.
        colorbar : bool
            If True, add a colorbar to the plot.
        colorbar_fontsize, fontsize : int
            Font sizes to use for colorbar and axis labels.
        linewidth : int or float
            Line width for plot lines.
        **kwargs : dict
            Additional keyword arguments passed to imshow.
    
    Returns:
        out : object
            The resulting plot.
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    if fgax is not None:
        fg, ax = fgax
        plt.sca(ax)
    else:
        fg, ax = plt.subplots(figsize=(3.5, 2.5))  # Adjust the figure size to fit well in IEEE format
    
    if aspect is None:
        aspect = grid.size(1) / grid.size(0)

    axvals = np.linspace(grid.min().item(), grid.max().item(), 9)
    if rescale:
        axvals = np.concatenate([np.linspace(grid.min().item(), 0, 5), np.linspace(0, grid.max().item(), 5)[1:]])
        grid = rescale_map_for_viz(grid.clone())  

    out = plt.imshow(grid.detach().cpu().numpy(), cmap=cmap, **kwargs)
    
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=linewidth)
    
    if colorbar:
        cbar = plt.colorbar(ax=ax)
        axlbls = [f'{v.item():.2f}' for v in axvals]
        cbar.set_ticks(np.linspace(-1, 1, 9))
        cbar.set_ticklabels(axlbls)
        cbar.ax.tick_params(labelsize=colorbar_fontsize, width=linewidth)  
    
    plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')

    return out

def plot_image_with_colorbar(image_path, data_array, xlabel='', ylabel='', title='',
                             colorbar=False, cmap='viridis', colorbar_fontsize=8, 
                             fontsize=8, save_path=None, **kwargs):
    """
    Plot an image (PNG) and utilize a separate data array to set axis range and colorbar.

    Parameters:
        image_path : str
            Path to the PNG image to be plotted.
        data_array : array-like
            Data array used for deriving axis range and colorbar.
        xlabel, ylabel, title : str
            Labels and title for the plot.
        colorbar : bool
            If True, add a colorbar to the plot using data from data_array.
        cmap : str
            Colormap to use for the colorbar.
        colorbar_fontsize, fontsize : int
            Font sizes to use for colorbar and axis labels.
        save_path : str or None
            Path including filename to save the plot as a PDF file.
            If None, the plot is not saved.
        **kwargs : dict
            Additional keyword arguments passed to imshow.
    
    Returns:
        out : object
            The resulting plot.
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Adjust the figure size

    # Load and display the image
    img = mpimg.imread(image_path)
    imgplot = ax.imshow(img, **kwargs)

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    # Add a colorbar based on the data array, if desired
    if colorbar:
        norm = Normalize(vmin=data_array.min(), vmax=data_array.max())
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)
    
    # Save the plot if a save_path is provided
    if save_path is not None:
        plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')
    
    return imgplot

def calculate_2d_histogram(data, bins, extent):
    # Determine the size of each bin
    bin_size_x = (extent[1] - extent[0]) / bins
    bin_size_y = (extent[3] - extent[2]) / bins

    # Initialize the histogram array
    histogram = [[0 for _ in range(bins)] for _ in range(bins)]
    # Iterate over each data point
    for x, y in data:
        # Determine the bin for x
        if extent[0] <= x < extent[1]:
            bin_x = int((x - extent[0]) / bin_size_x)
        else:
            continue  # Skip values outside the extent

        # Determine the bin for y
        if extent[2] <= y < extent[3]:
            bin_y = int((y - extent[2]) / bin_size_y)
        else:
            continue  # Skip values outside the extent

        # Increment the corresponding bin
        histogram[bin_y][bin_x] += 1

    return np.array(histogram)

def plot_posterior_2d(Z, xidx=None, yidx=None, fgax=None, bins=None, r=2, extent=None,
                      cmap='Reds', interpolation = "gaussian", errors=None, **kwargs):
    if extent is None:
        extent = [-r, r, -r, r]

    if isinstance(Z, distrib.Distribution):
        Z = Z.sample()
    if Z.size(1) > 2:
        assert xidx is not None and yidx is not None, 'Must specify xidx and yidx'
        Z = Z[..., [xidx, yidx]]
    assert Z.size(1) == 2, 'Z must have 2 dimensions'

    if fgax is not None:
        fg, ax = fgax
        plt.sca(ax)

    if bins is not None:
        # hist, *other = np.histogram2d(*Z.cpu().t().numpy(), bins=bins,
        #                             range = np.array([[extent[0], extent[1]], [extent[2], extent[3]]]))
        
        hist = calculate_2d_histogram(Z, bins=bins,extent=extent)
        map_extent = [extent[0], extent[1], extent[3], extent[2]]

        # change extent to allign with the scatter                            
        return plt.imshow(hist, cmap=cmap, interpolation=interpolation, extent=map_extent, **kwargs)
    else: 
        # plot scatter, size and color depends on the error
        color_scaler = MinMaxScaler()
        normalized_errors_color = color_scaler.fit_transform(errors.reshape(-1, 1)).flatten()
        size_scaler = MinMaxScaler(feature_range=(1, 100))  # Adjust the range (10, 100) as needed
        normalized_errors_size = size_scaler.fit_transform(errors.reshape(-1, 1)).flatten()
        # scatter_plot = plt.scatter(*Z.cpu().t().numpy(), alpha=normalized_errors_color, c=normalized_errors_color, s=normalized_errors_size, cmap="magma", **kwargs)
        xy = Z.cpu().t().numpy()
        scatter_plot = plt.scatter(xy[0], xy[1], s=0.1, alpha=0.5, color="black")
        # Set the limits of the plot to the specified extent
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])

        return scatter_plot

def plot_reconstruction_ll(autoencoder, z, axis):
    """
    Plotting reconstruction of a given latent point z
    """
    trajectory = autoencoder.decode(z)
    # rebuild trajectory
    trajectory = build_trajectories(trajectory, autoencoder)
    trajectory.to_csv("data/trajectories/trajectory.csv")
    # Get longitude and latitude
    trajectory = trajectory[['longitude', 'latitude']]
    # Plotting
    axis.plot(trajectory['longitude'], trajectory['latitude'], color='blue', linewidth=2)

def plot_reconstruction_track(autoencoder, z, axis):
    """
    Plotting reconstruction of a given latent point z
    """
    trajectory = autoencoder.decode(z)
    # rebuild trajectory
    trajectory = build_trajectories(trajectory, autoencoder)
    # Get longitude and latitude
    trajectory = trajectory[['longitude', 'latitude']]
    # Plotting
    axis.plot(trajectory['longitude'], trajectory['latitude'], color='blue', linewidth=2)

def plot_reconstruction_altitude(autoencoder, z, axis):
    """
    Plotting reconstruction of a given latent point z
    """
    trajectory = autoencoder.decode(z)
    # rebuild trajectory
    trajectory = build_trajectories(trajectory, autoencoder)
    # Get longitude and latitude
    trajectory = trajectory[['longitude', 'latitude']]
    # Plotting
    axis.plot(trajectory['longitude'], trajectory['latitude'], color='blue', linewidth=2)

def plot_raw(autoencoder, xidx, yidx, raw_dim, base=None, n=100, r=2, extent=None, batch_size: int = 128,
                    device: Optional[str] = None, pbar: Optional[tqdm] = None, **kwargs):
    
    '''
    Plotting trajectories from latent space
    '''
    if extent is None:
        extent = [-r, r, -r, r]
    seq_len = autoencoder.dataset_params['seq_len']

    zgrid = generate_2d_latent_map(xidx, yidx, base=base, n=n, r=r, extent=extent)
    H, W, _ = zgrid.size()

    # returns shape (100, 800)  aka (n_trajectories, seq_len*nb_features)
    # recs = push_forward(autoencoder, zgrid.view(H*W, -1), 'decode', device=device, batch_size=batch_size, pbar=pbar)
    recs = autoencoder.decode(zgrid.view(H*W, -1))
    # Getting track and timestamp for reconstructed trajectories
    recs_track = recs.view(H , W, seq_len, 4) # (n_trajectories, seq_len, nb_features)
    # Select first and last dimension (track and timestamp)
    recs_track = recs_track[:, :, :, raw_dim] # (n_trajectories, seq_len, 2)
    # detach and convert to numpy
    recs_track = recs_track.detach().cpu()
    
    # Plotting responses
    # rmap = push_forward(autoencoder, recs, 'encode').view(H, W, -1)
    rmap = autoencoder.encode(recs).mean.view(H, W, -1)
    umap = rmap.sub(zgrid)[..., [xidx, yidx]]
    mmap = umap.norm(p=2,dim=-1)
    dmap = compute_divergence_2d(umap)
    cmap = compute_mean_curvature_2d(umap)

    plt.tight_layout();
    
    # Plot on grid of size (H, W),
    fig, ax = plt.subplots(H, W, figsize=(20, 20))
    # Start from bottom left to top right
    for i in range(H):  # Start from the bottom row and move upwards
        for j in range(W):
            # Plotting track
            ax[i, j].plot(recs_track[i, j, :], range(seq_len))
            # Remove axis
            ax[i, j].axis('off')

    plt.show()

    
    # dims = [xidx, yidx]

    im_kwargs = dict(aspect = 'auto', extent=extent)
    fg, axs = plt.subplots(1,4, figsize=(9,2.5),sharex=True, sharey=True)
    plot_map(mmap, fgax=(fg,axs[0]), rescale=False, colorbar=False, **im_kwargs);
    # plt.title('Magnitude of Response')
    # plt.ylabel(f'Dimension {dims[1]}')
    # plt.xlabel(f'Dimension {dims[0]}')

    # plot_map(dmap, fgax=(fg,axs[1]), cmap='seismic', colorbar=True, **im_kwargs);
    # plt.title('Divergence')
    # plt.xlabel(f'Dimension {dims[0]}')

    # plot_map(cmap, fgax=(fg,axs[2]), cmap='viridis', colorbar=True, **im_kwargs);
    # plt.title('Mean Curvature')
    # plt.xlabel(f'Dimension {dims[0]}') 
    
def plot_latlon(autoencoder: Autoencoder, xidx, yidx, base=None, n=64, r=2, extent=None, batch_size: int = 128,
                    device: Optional[str] = None, pbar: Optional[tqdm] = None, **kwargs):
    
    '''
    Plotting trajectories from latent space
    '''
    if extent is None:
        extent = [-r, r, -r, r]
        
    zgrid = generate_2d_latent_map(xidx, yidx, base=base, n=n, r=r, extent=extent)
    H, W, _ = zgrid.size()

    # returns shape (100, 800)  aka (n_trajectories, seq_len*nb_features)
    # recs = push_forward(autoencoder, zgrid.view(H*W, -1), 'decode', device=device, batch_size=batch_size, pbar=pbar)
    recs = autoencoder.decode(zgrid.view(H*W, -1))
    
    # Rebuilding trajectories from track and groundspeed to altitude and longitude
    print("Longitudinal and Latitudinal trajectories rebuilt")
    recs_rb = build_trajectories(recs, autoencoder) 
    # save as csv
    recs_rb.to_csv("data/trajectories/trajectories_fcvae.csv")
    # Gettiong Longitude and Latitude from rebuilt trajectories
    recs_lonlat = recs_rb[['longitude', 'latitude']]
    # Get trajectory length
    seq_len = autoencoder.dataset_params['seq_len']
    # Reshape to (n_trajectories, seq_len, 2)
    recs_lonlat = torch.tensor(recs_lonlat.values).view(-1, seq_len, 2)
    # Plotting 

    lonlat_plot = plot_trajectories(recs_lonlat, mark_ends=False,on_map=False, **kwargs)

    return lonlat_plot

def rescale_map_for_viz(dmap):
    vmap = dmap.clone()
    sel = vmap > 0
    if sel.sum() > 0:
        vmap[sel] /= vmap[sel].max().abs()
    sel = vmap < 0
    if sel.sum()>0:
        vmap[sel] /= vmap[sel].min().abs()
    return vmap


def map_grid_list_to_coordinates(grid_points, n, r):
    """
    Map a list of (i, j) grid indices to (x, y) in [-r, r] coordinate system.
    
    Parameters:
    - grid_points: List of tuples. Each tuple is (i, j) grid index.
    - n: Integer. The size of the grid (n x n).
    - r: Float. The positive limit of the target coordinate system.
    
    Returns:
    List of tuples. Each tuple contains floats (x, y) mapped to the [-r, r] coordinate system.
    """
    mapped_coordinates = []
    
    for i, j in grid_points:
        # Ensure inputs are valid
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"Grid indices must be in range [0, {n-1}]")
        if r <= 0:
            raise ValueError("r must be a positive number")
        
        # Step 1: Normalize to [0, 1]
        i_prime = i / (n-1)
        j_prime = j / (n-1)
        
        # Step 2: Map to [-r, r]
        x = 2 * r * i_prime - r
        y = 2 * r * j_prime - r
        
        mapped_coordinates.append([x, y])
    
    return np.array(mapped_coordinates)

def plot_quiver_latex_corrected(data, xlabel='', ylabel='', title='', fgax=None, 
                                quiver_color='black', fontsize=11, linewidth=1, save_path='', **kwargs):
    """
    Plot a quiver map with customization options suitable for IEEE papers.

    Parameters:
        data : array-like
            Data to plot. Expected shape is (N, M, 2), where the last dimension stores vector components.
        xlabel, ylabel, title : str
            Labels and title for the plot.
        fgax : tuple
            (figure, axis) objects (if None, new objects are created).
        quiver_color : str
            Color of the quiver arrows.
        fontsize : int
            Font size to use for axis labels and title.
        linewidth : int or float
            Line width for plot lines.
        save_path : str
            Path to save the plot. If provided, the plot will be saved to this path.
        **kwargs : dict
            Additional keyword arguments passed to quiver.
    
    Returns:
        fg, ax : object
            The resulting plot figure and axis.
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    
    if fgax is not None:
        fg, ax = fgax
        plt.sca(ax)
    else:
        fg, ax = plt.subplots(figsize=(3.5, 2.5))  # Adjust the figure size to fit well in IEEE format

    # Get U and V components of vectors from data
    U = data[:, :, 0]
    V = data[:, :, 1]
    
    # Create a meshgrid for plotting
    X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
    
    # Quiver plot
    ax.quiver(X, Y, U, V, color=quiver_color, **kwargs)
    
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=linewidth)
    
    # Inverting the Y-axis for conventional Cartesian coordinates
    ax.invert_yaxis()
    
    # If save_path is provided, save the plot
    if save_path:
        plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')

    return fg, ax
