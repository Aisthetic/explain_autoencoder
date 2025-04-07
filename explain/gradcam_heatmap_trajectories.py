from math import e
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import torch
from tqdm import tqdm
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

# Import your custom modules (ensure these are in your PYTHONPATH or current directory)
from deep_traffic_generation.cvae import CVAE
from deep_traffic_generation.core.datasets import TSDataset
from explain.gradcam import GradCAM

###############################################################################
# 1) LOAD DATA AND MODEL
###############################################################################

def load_data_and_model(positions_path, labels_path, model_checkpoint_path, target_layer, device='cpu'):
    """
    Loads the positions, labels, dataset, model, and Grad-CAM instance.

    Args:
        positions_path (str): Path to the positions .npy file.
        labels_path (str): Path to the labels .npy file.
        model_checkpoint_path (str): Path to the model checkpoint .ckpt file.
        target_layer (str): Name of the target layer in the model for Grad-CAM.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        dataset: The dataset object.
        labels: Numpy array of labels.
        unique_labels: Numpy array of unique labels.
        model: Loaded CVAE model.
        gcam: Grad-CAM object.
    """
    # Load positions and labels
    positions = np.load(positions_path, allow_pickle=True)
    # positions.shape: (num_samples, seq_len, num_features)
    # Swap axes for shape: (num_samples, num_features, seq_len)
    positions = np.swapaxes(positions, 1, 2)
    print(f"Positions shape: {positions.shape}")

    labels = np.load(labels_path, allow_pickle=True)
    print(f"Labels shape: {labels.shape}")

    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")

    # Create dataset
    X = positions  # (num_samples, num_features, seq_len)
    y = None
    dataset = TSDataset(X, y)
    # Load model
    model = CVAE.load_from_checkpoint(
        model_checkpoint_path,
        dataset_params=dataset.parameters,
        map_location=device
    ).eval().to(device)

    # Create Grad-CAM object
    gcam = GradCAM(
        model,
        target_layer=target_layer,
        image_size=model.dataset_params["seq_len"],
        device=device
    )

    return dataset, labels, unique_labels, model, gcam


###############################################################################
# 2) COMPUTE GRAD-CAM ACTIVATIONS PER SAMPLE
###############################################################################

def compute_gradcam_per_sample(dataset, labels, model, gcam, device):
    """
    Computes Grad-CAM activations and latent vectors for all samples in the dataset.
    Normalizes Grad-CAM activations across all samples, for each latent dimension separately.
    Also collects the positions per sample.
    """
    gradcam_activations = []  # List to store Grad-CAM activations per sample
    sample_labels = []        # List to store labels per sample
    mu_list = []              # List to store latent vectors per sample
    positions_list = []       # List to store positions per sample

    print("Computing Grad-CAM activations for all samples...")
    for i in tqdm(range(len(dataset)), desc='Grad-CAM Computation'):
        x = dataset[i][0].to(device)
        # x.shape: (num_features, seq_len)
        _, mu, x_hat = model(x.unsqueeze(0))
        # x.unsqueeze(0).shape: (1, num_features, seq_len)
        # mu.shape: (1, latent_dim)
        # x_hat.shape: (1, num_features, seq_len)
        x_hat = x_hat.squeeze(0)
        # x_hat.shape: (num_features, seq_len)
        mu_list.append(mu.detach().cpu().numpy().squeeze())
        # mu_list[-1].shape: (latent_dim,)
        model.zero_grad()
        gcam_map = gcam.generate_all(mu).squeeze(1)
        # gcam_map.shape: (num_latent_layers, seq_len)
        gradcam_map_abs = torch.abs(gcam_map).detach().cpu().numpy()
        # gradcam_map_abs.shape: (num_latent_layers, seq_len)

        gradcam_activations.append(gradcam_map_abs.transpose(1, 0))  # Shape: (seq_len, num_latent_layers)
        sample_labels.append(labels[i])

        # Collect positions (use original x)
        positions = x.detach().cpu().numpy().T  # Shape: (seq_len, num_features)
        positions_list.append(positions)

    gradcam_activations = np.array(gradcam_activations)  # Shape: (num_samples, seq_len, num_latent_layers)
    positions_array = np.array(positions_list)  # Shape: (num_samples, seq_len, num_features)

    sample_labels = np.array(sample_labels)
    # sample_labels.shape: (num_samples,)
    mu_list = np.array(mu_list)
    # mu_list.shape: (num_samples, latent_dim)

    return gradcam_activations, sample_labels, mu_list, positions_array

###############################################################################
# 3) PCA ON LATENT VECTORS
###############################################################################

def perform_pca_on_latent_vectors(mu_list, n_components=2):
    """
    Performs PCA on the latent vectors and returns the PCA object.

    Returns:
        pca: PCA object
        scores: shape (num_samples, n_components)
    """
    print(f"Performing PCA on latent vectors with {n_components} components...")
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(mu_list)
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Explained variance: {pca.explained_variance_}")
    print(f"Components shape: {pca.components_.shape}")
    return pca, scores


###############################################################################
# 4) PROJECT GRAD-CAM ACTIVATIONS ONTO PCA COMPONENTS
###############################################################################

def compute_projected_gradcam_activations(gradcam_activations, pca):
    """
    Projects the Grad-CAM activations (over latent_dim) onto the PCA components.

    Args:
        gradcam_activations: (num_samples, seq_len, latent_dim)
        pca: PCA object

    Returns:
        gradcam_projections: (num_samples, seq_len, n_components)
    """
    num_samples, seq_len, latent_dim = gradcam_activations.shape
    print(f"Grad-CAM activations shape before projection: {gradcam_activations.shape}")

    # Flatten to (num_samples * seq_len, latent_dim)
    gradcam_flat = gradcam_activations.reshape(-1, latent_dim)
    # Multiply by PCA components
    gradcam_proj_flat = gradcam_flat @ pca.components_.T  # (num_samples*seq_len, n_components)
    gradcam_projections = gradcam_proj_flat.reshape(num_samples, seq_len, -1)

    print(f"Grad-CAM projections shape: {gradcam_projections.shape}")
    return gradcam_projections


###############################################################################
# 5) ORCHESTRATION: COMPUTE GRAD-CAM & PCA
###############################################################################

def compute_gradcam_and_pca(positions_path, labels_path, model_checkpoint_path,
                            target_layer, device='cpu', n_components=2):
    """
    Orchestrates:
      - Load data/model
      - Compute Grad-CAM for each sample
      - Perform PCA on latent vectors
      - Project Grad-CAM onto PCA components

    Returns:
      dataset, labels, unique_labels, model,
      gradcam_projections, pca,
      positions_array, sample_labels
    """
    dataset, labels, unique_labels, model, gcam = load_data_and_model(
        positions_path, labels_path, model_checkpoint_path, target_layer, device
    )

    gradcam_activations, sample_labels, mu_list, positions_array = compute_gradcam_per_sample(
        dataset, labels, model, gcam, device
    )

    pca, scores = perform_pca_on_latent_vectors(mu_list, n_components=n_components)
    gradcam_projections = compute_projected_gradcam_activations(gradcam_activations, pca)

    return (dataset, labels, unique_labels, model,
            gradcam_projections, pca,
            positions_array, sample_labels)


###############################################################################
# 6) COMPUTE RAW HEATMAPS (PER LABEL, PER COMPONENT) + GLOBAL MIN/MAX
###############################################################################

def compute_binned_heatmaps_per_label_and_component(
    positions_array,
    gradcam_projections,
    sample_labels,
    unique_labels,
    n_components,
    grid_size=100,
    bin_stats='mean',
    sigma=5,
    norm=True
):
    """
    Computes 2D binned (x,y) heatmaps of Grad-CAM intensities for each (label, PCA component).
    No per-label normalization is done here, so we can apply a global scale later.

    Returns:
        heatmaps_dict: dict with keys (label, comp_idx) -> 2D array (smoothed heatmap)
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        global_min: shape (n_components,), per-component minimum
        global_max: shape (n_components,), per-component maximum
    """
    from scipy.stats import binned_statistic_2d
    from scipy.ndimage import gaussian_filter

    num_samples, seq_len, n_comps = gradcam_projections.shape
    # We assume positions_array has shape (num_samples, seq_len, 2) or more.
    # We'll use the first two columns as (x, y).

    # 1) Find global x,y min/max
    all_positions_flat = positions_array.reshape(-1, positions_array.shape[-1])
    # Use the first 2 features as x,y
    global_x_min, global_x_max = np.nanmin(all_positions_flat[:, 0]), np.nanmax(all_positions_flat[:, 0])
    global_y_min, global_y_max = np.nanmin(all_positions_flat[:, 1]), np.nanmax(all_positions_flat[:, 1])

    # 2) Prepare dictionary and global min/max
    heatmaps_dict = {}
    gradcam_abs = np.abs(gradcam_projections)

    global_min = np.full(n_comps, np.inf)
    global_max = np.full(n_comps, -np.inf)

    # 3) For each label, for each component
    for comp_idx in range(n_components):
        for label in unique_labels:
            label_indices = np.where(sample_labels == label)[0]
            if len(label_indices) == 0:
                continue

            # Extract positions for these samples
            positions_label = positions_array[label_indices]  # (num_samples_label, seq_len, num_features)
            # Extract Grad-CAM for this component
            gradcam_label = gradcam_abs[label_indices, :, comp_idx]  # (num_samples_label, seq_len)

            # Flatten
            pos_flat = positions_label.reshape(-1, positions_label.shape[-1])  # (num_samples_label * seq_len, num_features)
            grad_flat = gradcam_label.reshape(-1)

            # Remove invalid
            valid_mask = (
                ~np.isnan(pos_flat).any(axis=1) &
                ~np.isnan(grad_flat) &
                np.isfinite(grad_flat)
            )
            pos_valid = pos_flat[valid_mask]
            grad_valid = grad_flat[valid_mask]
            if pos_valid.shape[0] == 0:
                continue

            x_valid = pos_valid[:, 0]
            y_valid = pos_valid[:, 1]

            # 2D binning
            heatmap, x_edges, y_edges, _ = binned_statistic_2d(
                x_valid, y_valid, grad_valid,
                statistic=bin_stats,
                bins=grid_size,
                range=[[global_x_min, global_x_max],
                       [global_y_min, global_y_max]]
            )
            heatmap = np.nan_to_num(heatmap)
            heatmap_smoothed = gaussian_filter(heatmap, sigma=sigma)

            # Store in dict
            if norm and heatmap_smoothed.max() > 0:
                heatmap_smoothed = heatmap_smoothed / heatmap_smoothed.max()
                
            heatmaps_dict[(label, comp_idx)] = heatmap_smoothed

            # Update global min/max for this component
            hm_min = heatmap_smoothed.min()
            hm_max = heatmap_smoothed.max()
            if hm_min < global_min[comp_idx]:
                global_min[comp_idx] = hm_min
            if hm_max > global_max[comp_idx]:
                global_max[comp_idx] = hm_max

    return (heatmaps_dict,
            (global_x_min, global_x_max),
            (global_y_min, global_y_max),
            global_min, global_max)


###############################################################################
# 7) PLOT LABEL-BY-LABEL HEATMAPS (GLOBAL SCALE)
###############################################################################

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_heatmaps_per_label_and_component(
    heatmaps_dict,
    positions_array,
    sample_labels,
    unique_labels,
    n_components,
    x_range,
    y_range,
    global_min,
    global_max,
    invert_y_axis=True,
    comp_norm=True,
    vertical_layout=True,
    save_path=None
):
    """
    Plots each (label, PCA component) heatmap from `heatmaps_dict` on a single global scale
    per component. This ensures the same brightness for the same Grad-CAM intensity.

    Args:
        heatmaps_dict: dict {(label, comp_idx) -> 2D heatmap}
        positions_array: (num_samples, seq_len, num_features)
        sample_labels: (num_samples,)
        unique_labels: array of unique labels
        n_components: number of PCA components
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        global_min, global_max: each shape (n_components,)
        invert_y_axis: bool
        comp_norm: bool, per component normalization 
        vertical_layout: bool, if True, plots in a vertical layout, else in a horizontal layout
        save_path: if provided, the figure is saved to this path
    """

    (global_x_min, global_x_max) = x_range
    (global_y_min, global_y_max) = y_range

    num_labels = len(unique_labels)
    
    if vertical_layout:
        fig, axes = plt.subplots(n_components, num_labels,
                                 figsize=(4 * num_labels, 4 * n_components),
                                 squeeze=False)
    else:
        # Create a grid with num_labels rows and n_components columns,
        # then transpose so that axes[comp_idx, label_idx] still holds.
        fig, axes = plt.subplots(num_labels, n_components,
                                 figsize=(4 * n_components, 4 * num_labels),
                                 squeeze=False)
        axes = axes.T

    plt.subplots_adjust(wspace=0.01, hspace=0.05)

    # Plot each label+component using the same scale for that component
    for comp_idx in range(n_components):
        comp_min = global_min[comp_idx]
        comp_max = global_max[comp_idx]
        denom = max(comp_max - comp_min, 1e-8)

        for label_idx, label in enumerate(unique_labels):
            ax = axes[comp_idx, label_idx]

            # Plot raw trajectories first
            label_indices = np.where(sample_labels == label)[0]
            positions_label = positions_array[label_indices]
            for sample_pos in positions_label:
                x_positions = sample_pos[:, 0]
                y_positions = sample_pos[:, 1]
                ax.plot(
                    x_positions,
                    y_positions,
                    color='black',
                    alpha=0.05,
                    linewidth=0.1
                )

            # Retrieve heatmap
            heatmap_smoothed = heatmaps_dict.get((label, comp_idx), None)
            if heatmap_smoothed is None:
                continue

            # Normalize if requested
            if comp_norm:
                heatmap_norm = (heatmap_smoothed - comp_min) / denom
            else:
                heatmap_norm = heatmap_smoothed

            ax.imshow(
                heatmap_smoothed.T,
                extent=[global_x_min, global_x_max, global_y_min, global_y_max],
                origin='lower',
                cmap='YlOrRd',
                aspect='auto',
            )

            # Hide ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # For vertical layout, place titles and row labels within each subplot.
            if vertical_layout:
                if comp_idx == 0:
                    ax.set_title(f"Label: {label}", pad=5)
                if label_idx == 0:
                    ax.set_ylabel(f"PC {comp_idx+1} - Latitude", rotation=90, labelpad=3)
                if comp_idx == n_components - 1:
                    ax.set_xlabel('Longitude')
            else:
                # For horizontal layout, we do not set individual subplot labels
                # (we will add them later as figure-level annotations).
                if comp_idx == n_components - 1:
                    ax.set_xlabel('Longitude')

            if invert_y_axis:
                ax.invert_yaxis()
            
            ax.set_aspect('equal')

    plt.tight_layout()

    # When using horizontal layout, add global labels on the borders.
    if not vertical_layout:
        # Add column labels (top of each column)
        for label_idx, label in enumerate(unique_labels):
            # Use the top row axes to get the positions
            ax = axes[0, label_idx]
            pos = ax.get_position()
            fig.text(
                (pos.x0 + pos.x1) / 2,
                pos.y1 + 0.01,
                f"Label: {label}",
                ha="center",
                va="bottom"
            )

        # Add row labels (left of each row)
        for comp_idx in range(n_components):
            ax = axes[comp_idx, 0]
            pos = ax.get_position()
            fig.text(
                pos.x0 - 0.01,
                (pos.y0 + pos.y1) / 2,
                f"PC {comp_idx+1} - Latitude",
                ha="right",
                va="center",
                rotation=90
            )

        # Optionally, add a global x-axis label at the bottom
        fig.text(0.5, 0.01, "Longitude", ha="center", va="bottom")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()




###############################################################################
# 8) PLOT COMBINED HEATMAP (GLOBAL SCALE)
###############################################################################

def plot_all_trajectories_with_and_without_heatmaps(
    heatmaps_dict,
    positions_array,
    sample_labels,
    unique_labels,
    n_components,
    x_range,
    y_range,
    global_min,
    global_max,
    invert_y_axis=True,
    concat_fun=None,
    show_trajectories=True,
    show_heatmaps=True,
    show_background_trajectories=True,
    save_path=None
):
    """
    1) Left subplot: all trajectories color-coded by label (no heatmap).
    2) Next subplots: sum of label-based heatmaps for each component,
       on the same global brightness scale.

    Args:
        heatmaps_dict: dict {(label, comp_idx) -> 2D heatmap}
        positions_array: (num_samples, seq_len, num_features)
        sample_labels: (num_samples,)
        unique_labels: unique labels
        n_components: number of PCA components
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        global_min, global_max: shape (n_components,)
        invert_y_axis: bool
        show_trajectories: bool, whether to show the trajectory plot.
        show_heatmaps: bool, whether to show heatmaps.
        show_background_trajectories: bool, whether to show background trajectories in heatmaps.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    (global_x_min, global_x_max) = x_range
    (global_y_min, global_y_max) = y_range

    # Define a consistent color map for labels
    color_map = {
        0: '#984ea3',
        2: '#1f77b4',
        3: '#e41a1c',
        4: '#4daf4a',
    }

    num_plots = 1 + n_components if show_trajectories and show_heatmaps else max(1, n_components)
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))

    if num_plots == 1:
        axes = [axes]

    plot_index = 0

    # 1) Left subplot: all trajectories color-coded by label (if enabled)
    if show_trajectories:
        ax_no_heatmap = axes[plot_index]
        plot_index += 1
        for label in unique_labels:
            label_indices = np.where(sample_labels == label)[0]
            color = color_map.get(label, '#000000')
            for idx in label_indices:
                x_positions = positions_array[idx, :, 0]
                y_positions = positions_array[idx, :, 1]
                ax_no_heatmap.plot(
                    x_positions,
                    y_positions,
                    color=color,
                    alpha=0.1,
                    linewidth=0.5
                )
        ax_no_heatmap.set_title("All Trajectories")
        ax_no_heatmap.set_xticks([])
        ax_no_heatmap.set_yticks([])
        ax_no_heatmap.set_aspect('equal')  # For the trajectory plot

        if invert_y_axis:
            ax_no_heatmap.invert_yaxis()

    # 2) For each component, sum across all labels (if enabled)
    num_samples = positions_array.shape[0]
    if show_heatmaps:
        for comp_idx in range(n_components):
            ax = axes[plot_index]
            plot_index += 1

            # Plot all trajectories in background (if enabled)
            if show_background_trajectories:
                for i in range(num_samples):
                    ax.plot(
                        positions_array[i, :, 0],
                        positions_array[i, :, 1],
                        color='black',
                        alpha=0.05,
                        linewidth=0.1
                    )

            # Sum across labels
            combined_heatmap = None
            for label in unique_labels:
                hm = heatmaps_dict.get((label, comp_idx), None)
                if hm is not None:
                    if combined_heatmap is None:
                        combined_heatmap = hm.copy()
                    else:
                        if concat_fun is not None:
                            combined_heatmap = concat_fun(combined_heatmap, hm)
                        else:
                            combined_heatmap += hm

            if combined_heatmap is not None:
                ax.imshow(
                    combined_heatmap.T,
                    extent=[global_x_min, global_x_max, global_y_min, global_y_max],
                    origin='lower',
                    cmap='YlOrRd',
                    aspect='equal',
                )

            ax.set_title(f"Combined Heatmap - PC {comp_idx+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            if invert_y_axis:
                ax.invert_yaxis()

            ax.set_aspect('equal')  # For each heatmap subplot

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.tight_layout()
    plt.show()

def create_modified_dataset(dataset, labels, sample_idx, gradcam_pca_values, threshold, pca_component=0):
    """
    Creates a modified dataset by zeroing out points in each sample (with the same label as the reference sample)
    where the Grad-CAM PCA value (for the specified PCA component) of the reference sample is below the threshold.
    
    Args:
        dataset: TSDataset, where each sample is accessed as dataset[idx][0] (a tensor of shape (num_features, seq_len)).
        labels (np.array): Array of labels.
        sample_idx (int): Index of the reference sample.
        gradcam_pca_values (np.array): Grad-CAM PCA values for the reference sample (shape: [seq_len, n_components]).
        threshold (float): Threshold value for masking.
        pca_component (int): PCA component index to use for thresholding.
        
    Returns:
        modified_dataset (np.array): Array of modified samples for all samples with the same label.
        original_indices (np.array): Indices of samples with the same label as the reference sample.
    """
    selected_label = labels[sample_idx]
    original_indices = np.where(labels == selected_label)[0]
    modified_dataset = []
    # Create a mask using the reference sample's Grad-CAM PCA values on the selected component.
    important_mask = gradcam_pca_values[:, pca_component] >= threshold  # shape: (seq_len,)
    
    for idx in original_indices:
        # Get the original sample (as a numpy array); assumed shape: (num_features, seq_len)
        sample = dataset[idx][0].detach().cpu().numpy()
        # Zero out points along the sequence where the reference mask is False.
        # This applies the same mask for all samples with the same label.
        modified_sample = np.where(important_mask, sample, 0)
        modified_dataset.append(modified_sample)
    
    return np.array(modified_dataset), original_indices


def compute_latents_with_original_pca(model, dataset, original_pca, device="cpu"):
    """
    Computes latent vectors for each sample in the given dataset using the model,
    then projects these latents using the provided original PCA object.
    
    Args:
        model: The trained CVAE model.
        dataset: A collection (list/array) of samples. Each sample is expected to be 
                 a tensor (or convertible to tensor) of shape (num_features, seq_len).
        original_pca: The precomputed PCA object (fitted on the original dataset's latents).
        device (str): Device to use ('cpu' or 'cuda').
        
    Returns:
        projected_latents (np.array): PCA-transformed latent vectors, shape (num_samples, n_components).
        latents (np.array): The raw latent vectors from the model.
    """
    latents = []
    for sample in dataset:
        # Ensure the sample is a torch tensor (if not already)
        if not torch.is_tensor(sample):
            sample_tensor = torch.tensor(sample, dtype=torch.float32).to(device)
        else:
            sample_tensor = sample.to(device)
        sample_tensor = sample_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            _, latent, _ = model(sample_tensor)
        latents.append(latent.cpu().numpy().squeeze())
    latents = np.array(latents)
    projected_latents = original_pca.transform(latents)
    return projected_latents, latents


def compute_pca_distances(original_latents, modified_latents):
    """
    Computes the Euclidean distance between corresponding latent vectors from two sets.
    
    Args:
        original_latents (np.array): Array of original latent vectors (shape: (num_samples, n_components)).
        modified_latents (np.array): Array of modified latent vectors (shape: (num_samples, n_components)).
    
    Returns:
        distances (list): List of Euclidean distances for each pair of latent vectors.
    """
    from scipy.spatial.distance import euclidean
    distances = []
    for orig, mod in zip(original_latents, modified_latents):
        distances.append(euclidean(orig, mod))
    return distances
def compute_global_latent_distance(percentile, 
                                   gradcam_first_pc, 
                                   positions_array, 
                                   sample_labels, 
                                   model, 
                                   pca, 
                                   original_latents,
                                   selected_label,
                                   device='cpu',
                                   selected_dimension=0):
    """
    Applies a global pruning mask based on the specified percentile of the mean gradcam_first_pc 
    (averaged over samples with the given selected_label) to the positions_array,
    computes latent vectors for the pruned dataset using the provided model and original PCA,
    and then calculates the absolute differences on a selected latent dimension between
    the original latent vectors and the pruned ones for each sample of the selected label.
    
    Args:
        percentile (float): The percentile (e.g., 50 for the 50th percentile) used to 
                            compute the threshold value from the mean gradcam_first_pc.
        gradcam_first_pc (np.ndarray): Global projected Grad-CAM activations on the first PCA component,
                                       shape (num_samples, seq_len).
        positions_array (np.ndarray): Original positions array, shape (num_samples, seq_len, num_features).
        sample_labels (array-like): Array of labels for each sample.
        model: Trained CVAE model.
        pca: PCA object fitted on the original dataset’s latent vectors.
        original_latents (np.ndarray): The latent vectors for the original dataset, shape (num_samples, latent_dim).
        selected_label: The label (from unique_labels) for which to compute the distances.
        device (str): 'cpu' or 'cuda'.
        selected_dimension (int): The index of the latent dimension to use for distance calculation.
        
    Returns:
        mean_distance (float): Mean absolute difference on the selected latent dimension 
                               between original and pruned latent vectors for samples with the selected label.
        distances (list): List of absolute differences for each sample (only for the selected label).
    """
    # Filter indices for the selected label
    label_indices = np.where(sample_labels == selected_label)[0]
    if len(label_indices) == 0:
        raise ValueError(f"No samples found with label {selected_label}")

    # Filter inputs to only include samples with the selected label
    gradcam_first_pc_label = gradcam_first_pc[label_indices]  # shape: (num_label_samples, seq_len)
    positions_array_label = positions_array[label_indices]      # shape: (num_label_samples, seq_len, num_features)
    original_latents_label = original_latents[label_indices]      # shape: (num_label_samples, latent_dim)
    
    # --- Step 1: Apply the global pruning mask ---
    # Compute the mean of the gradcam heatmaps over the selected label
    mean_gradcam = np.mean(gradcam_first_pc_label, axis=0)  # shape: (seq_len,)
    # Compute the threshold value from the mean gradcam values using the specified percentile
    threshold_value = np.percentile(mean_gradcam, percentile)
    print(f"Threshold value (percentile {percentile}) from mean gradcam: {threshold_value}")
    
    # Create a mask from the mean gradcam (for each time step)
    mask_single = mean_gradcam >= threshold_value  # shape: (seq_len,)
    # Expand this mask to all samples with the selected label
    mask = np.tile(mask_single, (len(label_indices), 1))  # shape: (num_label_samples, seq_len)
    
    # Expand mask to match the number of features
    mask_expanded = np.expand_dims(mask, axis=-1)  # (num_label_samples, seq_len, 1)
    mask_expanded = np.repeat(mask_expanded, positions_array_label.shape[2], axis=-1)  # (num_label_samples, seq_len, num_features)
    
    # Apply mask: set positions where mask is False to -1
    pruned_positions_array = np.where(mask_expanded, positions_array_label, -1)
    
    # --- Step 2: Compute latent vectors for the pruned dataset ---
    # Convert pruned_positions_array into a list of samples with shape (num_features, seq_len)
    pruned_dataset = [pruned_positions_array[i].T for i in range(pruned_positions_array.shape[0])]
    
    # Compute latent vectors using the helper function (assumed to work with a list of samples)
    pruned_latents, _ = compute_latents_with_original_pca(model, pruned_dataset, original_pca=pca, device=device)
    
    # --- Step 3: Compute distances on the selected latent dimension ---
    differences = [abs(orig[selected_dimension] - pruned[selected_dimension]) 
                   for orig, pruned in zip(original_latents_label, pruned_latents)]
    mean_distance = np.mean(differences)
    
    return mean_distance, differences



