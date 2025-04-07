# cvae_analysis.py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from deep_traffic_generation.core.datasets import TSDataset
from deep_traffic_generation.cvae import CVAE
from explain.gradcam import GradCAM
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import euclidean
import pandas as pd

# ---------------------------
# Data & Model Loading
# ---------------------------
def load_dataset(positions_path, labels_path):
    """
    Loads the dataset from provided numpy files and creates a TSDataset.

    Args:
        positions_path (str): Path to the positions npy file.
        labels_path (str): Path to the labels npy file.

    Returns:
        dataset (TSDataset): The dataset object.
        positions (np.array): Loaded positions (after axis swap).
        labels (np.array): Loaded labels.
    """
    positions = np.load(positions_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    positions = np.swapaxes(positions, 1, 2)
    X = positions
    y = None
    dataset = TSDataset(X, y)
    return dataset, positions, labels

def load_cvae_model(model_path, dataset_params, device="cuda"):
    """
    Loads a pretrained CVAE model from a checkpoint.

    Args:
        model_path (str): Path to the model checkpoint.
        dataset_params (dict): Parameters from the dataset.
        device (str): Device to load the model on ("cpu" or "cuda").

    Returns:
        model: The loaded CVAE model.
    """
    model = CVAE.load_from_checkpoint(
        model_path,
        dataset_params=dataset_params,
        map_location=device
    ).eval().to(device)
    return model

def init_gradcam(model, target_layer, image_size, device="cuda"):
    """
    Initializes a GradCAM object for the given model and target layer.

    Args:
        model: The CVAE model.
        target_layer (str): The layer name on which to target GradCAM.
        image_size (int): The sequence length (or image size).
        device (str): "cpu" or "cuda".

    Returns:
        gcam: The initialized GradCAM object.
    """
    gcam = GradCAM(
        model,
        target_layer=target_layer,
        image_size=image_size,
        device=device
    )
    return gcam

# ---------------------------
# Reconstructions & PCA
# ---------------------------
def compute_reconstructions_and_latents(model, dataset, device="cuda"):
    """
    Computes the reconstructions and latent vectors for each sample in the dataset.

    Args:
        model: The CVAE model.
        dataset: The TSDataset.
        device (str): "cpu" or "cuda".

    Returns:
        reconstructions (np.array): Model reconstructions.
        latents (np.array): Latent vectors.
    """
    reconstructions = []
    latents = []
    for i in tqdm(range(len(dataset))):
        x = dataset[i][0].unsqueeze(0).to(device)
        _, latent, reconstruction = model(x)
        reconstructions.append(reconstruction.cpu().detach().numpy())
        latents.append(latent.cpu().detach().numpy())
    return np.array(reconstructions), np.array(latents)

def compute_latents_with_original_pca(model, dataset, original_pca, device="cpu"):
    """
    Computes latent vectors for the provided dataset and projects them using the given original PCA.
    
    Args:
        model: The CVAE model.
        dataset: A list/array of samples. Each sample is expected to be in the format 
                 (num_features, seq_len) as a tensor or numpy array.
        original_pca: A precomputed PCA object (e.g., fitted on the original dataset's latents).
        device (str): "cpu" or "cuda".
        
    Returns:
        projected_latents: PCA-transformed latent vectors computed via original_pca.transform().
        latents: The raw latent vectors for each sample.
    """
    latents = []
    for sample in dataset:
        # Ensure the sample is a torch tensor
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample)
        x = sample.to(device).unsqueeze(0)  # Add batch dimension
        _, latent, _ = model(x)
        latents.append(latent.cpu().detach().numpy())
    latents = np.array(latents).squeeze()  # shape: (num_samples, latent_dim)
    projected_latents = original_pca.transform(latents)
    return projected_latents, latents


def plot_modified_trajectories(original_dataset, modified_dataset, title="Modified Dataset Trajectories", invert_y_axis=True):
    """
    Plots trajectories of the original dataset, but uses the modified dataset to determine which points
    are kept (nonzero) versus masked (zero). Kept points are plotted in green and masked points in red.
    
    Each sample in both datasets is expected to be an array of shape (num_features, seq_len).
    The modified dataset should have unimportant points replaced with 0.
    
    Args:
        original_dataset (np.array): Array of original samples (used for plotting).
        modified_dataset (np.array): Array of modified samples (used to determine the mask).
        title (str): Title of the plot.
        invert_y_axis (bool): Whether to invert the Y-axis.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    
    first_sample = True  # for legend labels only once
    
    for orig_sample, mod_sample in zip(original_dataset, modified_dataset):
        orig_sample = np.array(orig_sample)
        mod_sample = np.array(mod_sample)
        # Create mask: a point is "kept" if any coordinate in the modified sample is nonzero
        kept_mask = np.any(mod_sample != 0, axis=0)
        masked_mask = ~kept_mask
        
        # Use the original sample's points for plotting
        x_kept = orig_sample[0, kept_mask]
        y_kept = orig_sample[1, kept_mask]
        x_masked = orig_sample[0, masked_mask]
        y_masked = orig_sample[1, masked_mask]
        
        if first_sample:
            plt.scatter(x_kept, y_kept, color='green', s=1, label='Kept', alpha=0.1)
            plt.scatter(x_masked, y_masked, color='red', s=1, label='Masked')
            first_sample = False
        else:
            plt.scatter(x_kept, y_kept, color='green', s=1)
            plt.scatter(x_masked, y_masked, color='red', s=1)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    if invert_y_axis:
        plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()



def apply_pca(latents, n_components=2):
    """
    Applies PCA to reduce the dimensionality of latent vectors.

    Args:
        latents (np.array): Latent vectors.
        n_components (int): Number of PCA components.

    Returns:
        pca: Fitted PCA model.
        pca_latents (np.array): PCA-transformed latent vectors.
    """
    latents = np.array(latents).squeeze()
    pca = PCA(n_components=n_components)
    pca_latents = pca.fit_transform(latents)
    return pca, pca_latents

# ---------------------------
# Visualization Functions
# ---------------------------
def plot_trajectory_with_gradcam(
    model, dataset, gcam, pca, pca_latents, labels, sample_idx=None,
    n_components=1, device="cpu", invert_y_axis=True, save_path=None
):
    """
    Visualizes a sample's trajectory with Grad-CAM overlay and shows its position in the PCA latent space.

    Args:
        model: CVAE model.
        dataset: TSDataset.
        gcam: GradCAM object.
        pca: Fitted PCA model.
        pca_latents (np.array): PCA-transformed latent vectors.
        labels (np.array): Labels corresponding to each sample.
        sample_idx (int, optional): Index of sample to visualize. If None, a random sample is chosen.
        n_components (int): Number of PCA components to use.
        device (str): "cpu" or "cuda".
        invert_y_axis (bool): Whether to invert the y-axis in trajectory plots.
        save_path (str, optional): Path to save the plot.
    """
    if sample_idx is None:
        sample_idx = np.random.randint(0, len(dataset))
    
    x = dataset[sample_idx][0].to(device)
    _, mu, x_hat = model(x.unsqueeze(0))
    x_hat = x_hat.squeeze(0)
    
    model.zero_grad()
    gcam_map = gcam.generate_all(mu).squeeze(1)  # (num_latent_layers, seq_len)
    gradcam_map_abs = np.abs(gcam_map.detach().cpu().numpy().transpose(1, 0))  # (seq_len, num_latent_layers)
    x_np = x.detach().cpu().numpy()
    
    n_components = min(n_components, pca.components_.shape[0])
    gradcam_pca = gradcam_map_abs @ pca.components_[:n_components].T
    gradcam_norm = (gradcam_pca - gradcam_pca.min(axis=0)) / (gradcam_pca.max(axis=0) - gradcam_pca.min(axis=0))
    
    correction = 2
    plt.rcParams.update({
        "text.usetex": False,  
        "font.family": "serif",
        "font.size": 10 + correction,
        "axes.titlesize": 11 + correction,
        "axes.labelsize": 10 + correction,
        "xtick.labelsize": 9 + correction,
        "ytick.labelsize": 9 + correction,
        "legend.fontsize": 9 + correction,
    })
    
    color_map = {
        0: '#000000',
        '(': '#1f77b4',
        ')': '#e41a1c',
        '\Omega': '#4daf4a',
    }
    
    fig, axes = plt.subplots(n_components + 1, 1, figsize=(8, 6 * (n_components + 1)), sharex=False)
    if n_components == 1:
        axes = [axes]
    
    # PCA scatter plot
    ax_pca = axes[0]
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = np.where(labels == label)
        color = color_map.get(label, '#000000')
        ax_pca.scatter(pca_latents[idx, 0], pca_latents[idx, 1], color=color, label=label, alpha=1, s=5)
    ax_pca.scatter(pca_latents[sample_idx, 0], pca_latents[sample_idx, 1],
                   s=200, color="red", edgecolors="black", label="Selected Sample")
    ax_pca.set_xlabel("PCA Component 1")
    ax_pca.set_ylabel("PCA Component 2")
    ax_pca.set_title("PCA Projection of Latent Vectors")
    ax_pca.legend(title="Labels")
    ax_pca.grid(False)
    ax_pca.set_xticks([])
    ax_pca.set_yticks([])
    
    # Trajectory plots with Grad-CAM coloring
    for j, ax in enumerate(axes[1:]):
        for i in range(x_np.shape[1] - 1):
            ax.plot(x_np[0, i:i+2], x_np[1, i:i+2],
                    color=plt.cm.YlOrRd(gradcam_norm[i, j]), linewidth=1.5)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                   norm=plt.Normalize(vmin=gradcam_pca[:, j].min(), vmax=gradcam_pca[:, j].max()))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f"Grad-CAM Importance (PCA {j+1})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        if invert_y_axis:
            ax.invert_yaxis()
        ax.set_title(f"Trajectory Colored by Grad-CAM Importance (PCA {j+1})")
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Plotted sample index: {sample_idx}")

def compute_gradcam_maps(model, dataset, gcam, indices, device="cpu"):
    """
    Computes Grad-CAM maps for given dataset indices.

    Args:
        model: The CVAE model.
        dataset: TSDataset.
        gcam: GradCAM object.
        indices (list): List of sample indices.
        device (str): "cpu" or "cuda".

    Returns:
        List of Grad-CAM maps (each as a numpy array).
    """
    gradcam_maps = []
    for idx in indices:
        x = dataset[idx][0].to(device)
        _, mu, _ = model(x.unsqueeze(0))
        model.zero_grad()
        gcam_map = gcam.generate_all(mu).squeeze(1)
        gradcam_map_abs = np.abs(gcam_map.detach().cpu().numpy().transpose(1, 0))
        gradcam_maps.append(gradcam_map_abs)
    return gradcam_maps

def plot_sample_with_heatmap_and_pca(
    model, dataset, gcam, pca, pca_latents, labels, sample_idx,
    pca_component=0, grid_size=100, sigma=5, device="cpu",
    invert_y_axis=True, save_path=None
):
    """
    For a single sample, computes its Grad-CAM map, projects it onto PCA components,
    computes a heatmap, and creates a two-panel plot:
      - Left: PCA scatter plot with only the related label colored.
      - Right: Trajectory with an overlaid Grad-CAM heatmap.
    """
    # Compute Grad-CAM map for the sample
    gradcam_map = compute_gradcam_maps(model, dataset, gcam, [sample_idx], device)[0]
    gradcam_map_pca = gradcam_map @ pca.components_.T  # (seq_len, n_components)
    gradcam_component = gradcam_map_pca[:, pca_component]
    gradcam_component_norm = (gradcam_component - gradcam_component.min()) / (
        gradcam_component.max() - gradcam_component.min() + 1e-8
    )
    
    # Retrieve the sample's trajectory (using first two features as x and y)
    x = dataset[sample_idx][0].to(device)
    x_np = x.detach().cpu().numpy()
    positions = x_np.T  # (seq_len, num_features)
    positions_xy = positions[:, :2]
    
    # Compute the binned and blurred heatmap for this sample
    heatmap, x_range, y_range = compute_single_heatmap(
        positions_xy, gradcam_component_norm, grid_size=grid_size, sigma=sigma, norm=True
    )
    
    # Create a two-panel plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ---- Left Subplot: PCA Scatter Plot ----
    ax_pca = axes[0]
    # Get the target label and indices
    target_label = labels[sample_idx]
    target_idx = np.where(labels == target_label)[0]
    other_idx = np.where(labels != target_label)[0]
    
    # Define color mapping
    color_map = {
        0: '#000000',
        '(': '#1f77b4',
        ')': '#e41a1c',
        '\Omega': '#4daf4a',
    }
    
    # Plot target samples in color and others in black
    ax_pca.scatter(
        pca_latents[target_idx, 0],
        pca_latents[target_idx, 1],
        color=color_map.get(target_label, 'black'),
        label=target_label, alpha=1, s=5
    )
    ax_pca.scatter(
        pca_latents[other_idx, 0],
        pca_latents[other_idx, 1],
        color='black',
        label='Other', alpha=0.5, s=5
    )
    ax_pca.scatter(
        pca_latents[sample_idx, 0],
        pca_latents[sample_idx, 1],
        s=200, color="red", edgecolors="black", label="Selected Sample"
    )
    ax_pca.set_xlabel("PCA Component 1")
    ax_pca.set_ylabel("PCA Component 2")
    ax_pca.set_title("PCA Projection of Latent Vectors")
    ax_pca.legend(title="Labels")
    ax_pca.grid(False)
    ax_pca.set_xticks([])
    ax_pca.set_yticks([])
    
    # ---- Right Subplot: Trajectory with Heatmap Overlay ----
    ax_traj = axes[1]
    # Plot the trajectory (faint black line)
    ax_traj.plot(
        positions_xy[:, 0],
        positions_xy[:, 1],
        color='black', alpha=0.3, linewidth=1
    )
    # Overlay the heatmap
    ax_traj.imshow(
        heatmap.T, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin='lower', cmap='YlOrRd', aspect='auto'
    )
    if invert_y_axis:
        ax_traj.invert_yaxis()
    
    ax_traj.set_title(f"Trajectory with Grad-CAM Heatmap (PCA Component {pca_component+1})")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------
# Heatmap & Modified Dataset Analysis
# ---------------------------
def compute_single_heatmap(positions, gradcam_values, grid_size=100, statistic='mean', sigma=5, norm=True):
    """
    Computes a 2D binned heatmap with a Gaussian blur.

    Args:
        positions (np.array): Array of shape (seq_len, 2) with x and y coordinates.
        gradcam_values (np.array): Grad-CAM values for each position.
        grid_size (int): Number of bins per axis.
        statistic (str): Binning statistic (e.g., 'mean').
        sigma (int): Sigma for Gaussian filter.
        norm (bool): Whether to normalize the heatmap.

    Returns:
        heatmap_smoothed (np.array): The smoothed heatmap.
        x_range (tuple): (x_min, x_max) of the grid.
        y_range (tuple): (y_min, y_max) of the grid.
    """
    x = positions[:, 0]
    y = positions[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    heatmap, x_edges, y_edges, _ = binned_statistic_2d(
        x, y, gradcam_values,
        statistic=statistic,
        bins=grid_size,
        range=[[x_min, x_max], [y_min, y_max]]
    )
    heatmap = np.nan_to_num(heatmap)
    heatmap_smoothed = gaussian_filter(heatmap, sigma=sigma)
    if norm and heatmap_smoothed.max() > 0:
        heatmap_smoothed = heatmap_smoothed / heatmap_smoothed.max()
    return heatmap_smoothed, (x_min, x_max), (y_min, y_max)

def compute_gradcam_pca(model, dataset, gcam, sample_idx, pca, device="cpu"):
    """
    Computes the Grad-CAM map for a sample and projects it onto PCA space.

    Args:
        model: CVAE model.
        dataset: TSDataset.
        gcam: GradCAM object.
        sample_idx (int): Index of the sample.
        pca: Fitted PCA model.
        device (str): "cpu" or "cuda".

    Returns:
        gradcam_pca_values (np.array): Grad-CAM values projected onto PCA space.
    """
    x = dataset[sample_idx][0].to(device)
    _, mu, _ = model(x.unsqueeze(0))
    model.zero_grad()
    gcam_map = gcam.generate_all(mu).squeeze(1).detach().cpu().numpy().T
    gradcam_pca_values = gcam_map @ pca.components_.T
    return gradcam_pca_values

def create_modified_dataset(dataset, labels, sample_idx, gradcam_pca_values, threshold, pca_component=0):
    """
    Creates a modified dataset where only important points (based on the Grad-CAM values for a specific PCA component)
    are kept, and the rest are replaced by -1.

    Args:
        dataset: Original dataset.
        labels: Corresponding labels for each sample.
        sample_idx: Index of the reference sample.
        gradcam_pca_values: Grad-CAM PCA values for the selected sample (shape: [seq_len, n_components]).
        threshold: Threshold for selecting important points.
        pca_component: The index of the PCA component to use for thresholding (default is 0).

    Returns:
        modified_dataset: Dataset with only important points retained.
        original_indices: Indices of samples with the same label as the reference sample.
    """
    selected_label = labels[sample_idx]
    original_indices = np.where(labels == selected_label)[0]
    # Use only the specified PCA component for thresholding
    important_points = gradcam_pca_values[:, pca_component] > threshold
    modified_dataset = []
    for idx in original_indices:
        sample = dataset[idx][0].detach().cpu().numpy()
        modified_sample = np.full_like(sample, fill_value=-1)
        modified_sample[:, important_points] = sample[:, important_points]
        modified_dataset.append(modified_sample)
    return np.array(modified_dataset), original_indices


def compute_latents_and_pca(model, dataset, device="cpu"):
    """
    Computes latent vectors for a dataset and applies PCA.

    Args:
        model: CVAE model.
        dataset: Dataset for which to compute latents.
        device (str): "cpu" or "cuda".

    Returns:
        pca_latents (np.array): PCA-transformed latent vectors.
        latents (np.array): Original latent vectors.
    """
    latents = []
    for sample in dataset:
        x = torch.tensor(sample).to(device).unsqueeze(0)
        _, latent, _ = model(x)
        latents.append(latent.cpu().detach().numpy())
    latents = np.array(latents).squeeze()
    pca = PCA(n_components=2)
    pca_latents = pca.fit_transform(latents)
    return pca_latents, latents

def compute_pca_distances(original_pca, modified_pca):
    """
    Computes Euclidean distances between original and modified PCA coordinates.

    Args:
        original_pca (np.array): PCA coordinates of original samples.
        modified_pca (np.array): PCA coordinates of modified samples.

    Returns:
        distances (list): Euclidean distances for each sample.
    """
    distances = [euclidean(original_pca[i], modified_pca[i]) for i in range(len(original_pca))]
    return distances

def save_results_to_csv(results_df, csv_path="pca_distance_analysis.csv"):
    """
    Saves a DataFrame of results to a CSV file.

    Args:
        results_df (pd.DataFrame): DataFrame containing results.
        csv_path (str): Destination file path.
    """
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    return csv_path

# ---------------------------
# Main Block (for testing)
# ---------------------------
if __name__ == "__main__":
    print("This module provides functions for CVAE analysis and visualization. Import it in your Jupyter notebook to use its functions.")
