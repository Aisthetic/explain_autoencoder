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

plt.rcParams.update({
    "text.usetex": False,          # Enable LaTeX rendering
    "font.family": "serif",       # Use serif fonts
    "font.size": 10,              # Base font size
    "axes.titlesize": 11,         # Titles slightly larger
    "axes.labelsize": 10,         # Axis labels size
    "xtick.labelsize": 9,         # Tick labels size
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


###############################################################################
#                      1) LOAD DATA AND MODEL                                 #
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
        model: Loaded model.
        gcam: Initialized Grad-CAM object.
    """
    # Load positions and labels
    positions = np.load(positions_path, allow_pickle=True)
    # If shape is (num_samples, seq_len, 1), swap axes to (num_samples, 1, seq_len)
    positions = np.swapaxes(positions, 1, 2)
    print(f"Positions shape after swapaxes: {positions.shape}")

    labels = np.load(labels_path, allow_pickle=True)
    print(f"Labels shape: {labels.shape}")

    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")

    # Create dataset (no explicit labels needed if CVAE is unsupervised)
    X = positions  # shape: (num_samples, 1, seq_len)
    y = None
    dataset = TSDataset(X, y)

    # Load CVAE model
    model = CVAE.load_from_checkpoint(
        model_checkpoint_path,
        dataset_params=dataset.parameters,
        map_location=device
    ).eval().to(device)

    # Create Grad-CAM instance
    gcam = GradCAM(
        model,
        target_layer=target_layer,
        image_size=model.dataset_params["seq_len"],
        device=device
    )

    return dataset, labels, unique_labels, model, gcam


###############################################################################
#                      2) COMPUTE GRAD-CAM ACTIVATIONS PER SAMPLE             #
###############################################################################

def compute_gradcam_per_sample(dataset, labels, model, gcam, device):
    """
    Computes Grad-CAM activations and latent vectors for all samples in the dataset.
    Also collects the raw 1D positions per sample.

    Returns:
        gradcam_activations: Numpy array, shape (num_samples, seq_len, latent_dim)
        sample_labels:       Numpy array of labels
        mu_list:            (num_samples, latent_dim)
        positions_array:    (num_samples, seq_len, 1)
    """
    gradcam_activations = []  # store Grad-CAM per sample
    sample_labels = []
    mu_list = []
    positions_list = []  # store 1D positions (shape: seq_len, 1)

    print("Computing Grad-CAM activations for all samples...")
    for i in tqdm(range(len(dataset)), desc='Grad-CAM Computation'):
        x = dataset[i][0].to(device)  # shape: (1, seq_len)
        # Forward through the model
        _, mu, x_hat = model(x.unsqueeze(0))  # x_hat: (1, 1, seq_len)
        x_hat = x_hat.squeeze(0)              # (1, seq_len)

        mu_list.append(mu.detach().cpu().numpy().squeeze())  # (latent_dim,)

        # Generate Grad-CAM
        model.zero_grad()
        gcam_map = gcam.generate_all(mu).squeeze(1)  # (latent_dim, seq_len)
        gradcam_map_abs = torch.abs(gcam_map).detach().cpu().numpy()

        # Transpose to (seq_len, latent_dim)
        gradcam_activations.append(gradcam_map_abs.transpose(1, 0))

        sample_labels.append(labels[i])

        # Collect positions
        positions = x.detach().cpu().numpy().T  # (seq_len, 1)
        positions_list.append(positions)

    gradcam_activations = np.array(gradcam_activations)  # (num_samples, seq_len, latent_dim)
    positions_array = np.array(positions_list)           # (num_samples, seq_len, 1)
    sample_labels = np.array(sample_labels)              # (num_samples,)
    mu_list = np.array(mu_list)                          # (num_samples, latent_dim)

    return gradcam_activations, sample_labels, mu_list, positions_array


###############################################################################
#                      3) PCA ON LATENT VECTORS                                #
###############################################################################

def perform_pca_on_latent_vectors(mu_list, n_components=2):
    """
    Performs PCA on the latent vectors and returns the PCA object.
    """
    print(f"Performing PCA on latent vectors with {n_components} components...")
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(mu_list)
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Explained variance by each component: {pca.explained_variance_}")
    print(f"Components shape: {pca.components_.shape}")
    return pca, scores


###############################################################################
#                      4) PROJECT GRAD-CAM ONTO PCA COMPONENTS                #
###############################################################################

def compute_projected_gradcam_activations(gradcam_activations, pca):
    """
    Projects the Grad-CAM activations (over latent_dim) onto the PCA components.

    Args:
        gradcam_activations: (num_samples, seq_len, latent_dim)
        pca: trained PCA object

    Returns:
        gradcam_projections: (num_samples, seq_len, n_components)
    """
    num_samples, seq_len, latent_dim = gradcam_activations.shape
    print(f"Grad-CAM activations shape before projection: {gradcam_activations.shape}")

    # Flatten to (num_samples * seq_len, latent_dim)
    gradcam_activations_flat = gradcam_activations.reshape(-1, latent_dim)

    # Project onto PCA components
    gradcam_projections_flat = gradcam_activations_flat @ pca.components_.T
    gradcam_projections = gradcam_projections_flat.reshape(num_samples, seq_len, -1)

    print(f"Grad-CAM projections shape: {gradcam_projections.shape}")
    return gradcam_projections


###############################################################################
#                      5) ORCHESTRATION: COMPUTE GRAD-CAM & PCA               #
###############################################################################

def compute_gradcam_and_pca(
    positions_path, labels_path, model_checkpoint_path,
    target_layer, device='cpu', n_components=2
):
    """
    Orchestrates:
        1) Load data/model
        2) Compute Grad-CAM for each sample
        3) Perform PCA on latent vectors
        4) Project Grad-CAM onto PCA components

    Returns:
        dataset, labels, unique_labels, model,
        gradcam_projections, pca,
        positions_array,
        sample_labels
    """
    # Load
    dataset, labels, unique_labels, model, gcam = load_data_and_model(
        positions_path, labels_path, model_checkpoint_path, target_layer, device
    )

    # Compute Grad-CAM
    gradcam_activations, sample_labels, mu_list, positions_array = compute_gradcam_per_sample(
        dataset, labels, model, gcam, device
    )

    # PCA on latent vectors
    pca, scores = perform_pca_on_latent_vectors(mu_list, n_components=n_components)

    # Project Grad-CAM onto principal components
    gradcam_projections = compute_projected_gradcam_activations(gradcam_activations, pca)

    return (
        dataset,
        labels,
        unique_labels,
        model,
        gradcam_projections,
        pca,
        positions_array,
        sample_labels
    )


###############################################################################
#          6) COMPUTE RAW HEATMAPS PER LABEL/COMPONENT + GLOBAL SCALE         #
###############################################################################

def compute_binned_heatmaps_per_label_and_component(
    positions_array,
    gradcam_projections,
    sample_labels,
    unique_labels,
    n_components,
    grid_size=200,
    bin_stats='mean',
    sigma=5,
    normalize=False
):
    """
    Computes 2D binned heatmaps (time x value) of Grad-CAM intensities,
    for each (label, PCA component). We do not normalize each heatmap
    separately, so that we can use a global brightness scale.

    Returns:
        heatmaps_dict: dict with keys (label, comp_idx) -> 2D array
        (val_min, val_max): range of 1D values across entire dataset
        global_min, global_max: each is a numpy array of shape (n_components,)
                                storing min/max across *all labels* for that component
    """
    from scipy.stats import binned_statistic_2d
    from scipy.ndimage import gaussian_filter

    num_samples, seq_len, n_comps = gradcam_projections.shape
    time_indices = np.arange(seq_len)

    # Flatten positions to find global min/max for the value axis
    value_all = positions_array.reshape(-1)
    val_min, val_max = np.nanmin(value_all), np.nanmax(value_all)

    # Dictionary to store each label+component heatmap
    heatmaps_dict = {}

    # Arrays to track global min and max for each component
    global_min = np.full(n_components, np.inf)
    global_max = np.full(n_components, -np.inf)

    gradcam_abs = np.abs(gradcam_projections)

    # Compute heatmap per (label, component)
    for comp_idx in range(n_components):
        for label in unique_labels:
            label_indices = np.where(sample_labels == label)[0]
            if len(label_indices) == 0:
                continue

            # Positions for this label
            pos_label = positions_array[label_indices, :, 0]    # shape: (num_label_samples, seq_len)
            grad_label = gradcam_abs[label_indices, :, comp_idx] # same shape

            # Flatten
            n_label_samples = len(label_indices)
            time_tile = np.tile(time_indices, n_label_samples)  # shape: (num_label_samples*seq_len,)

            val_flat = pos_label.flatten()
            grad_flat = grad_label.flatten()

            # Remove invalid
            valid_mask = (
                ~np.isnan(time_tile) &
                ~np.isnan(val_flat) &
                ~np.isnan(grad_flat) &
                np.isfinite(time_tile) &
                np.isfinite(val_flat) &
                np.isfinite(grad_flat)
            )
            if not np.any(valid_mask):
                continue

            time_valid = time_tile[valid_mask]
            val_valid  = val_flat[valid_mask]
            grad_valid = grad_flat[valid_mask]

            # Binning
            heatmap, x_edges, y_edges, _ = binned_statistic_2d(
                time_valid,
                val_valid,
                grad_valid,
                statistic=bin_stats,
                bins=[seq_len, grid_size],
                range=[[0, seq_len], [val_min, val_max]]
            )
            heatmap = np.nan_to_num(heatmap)
            heatmap_smoothed = gaussian_filter(heatmap, sigma=sigma)

            if normalize:
                heatmap_smoothed = (heatmap_smoothed - heatmap_smoothed.min()) / max(1e-8, heatmap_smoothed.max()- heatmap_smoothed.min())

            # Store raw (smoothed) heatmap
            heatmaps_dict[(label, comp_idx)] = heatmap_smoothed

            # Track global min/max for this component
            hm_min = heatmap_smoothed.min()
            hm_max = heatmap_smoothed.max()
            if hm_min < global_min[comp_idx]:
                global_min[comp_idx] = hm_min
            if hm_max > global_max[comp_idx]:
                global_max[comp_idx] = hm_max

    return heatmaps_dict, (val_min, val_max), (global_min, global_max)


###############################################################################
#    7) PLOT: LABEL-BY-LABEL USING THE SAME GLOBAL SCALE FOR EACH COMPONENT   #
###############################################################################
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmaps_per_label_and_component(
    heatmaps_dict,
    positions_array,
    sample_labels,
    unique_labels,
    n_components,
    val_range,
    global_extrema,
    invert_y_axis=False,
    save_path=None  # Optional path to save the figure
):
    """
    Plots each (label, component) heatmap from `heatmaps_dict` on a common
    global scale (i.e., same min/max) per component.

    Args:
        heatmaps_dict: Dictionary keyed by (label, comp_idx) -> 2D heatmap.
        positions_array: Array of shape (num_samples, seq_len, 1).
        sample_labels:   Array of shape (num_samples,).
        unique_labels:   Unique labels.
        n_components:    Number of PCA components.
        val_range:       Tuple (val_min, val_max) across the entire dataset.
        global_extrema:  Tuple (global_min, global_max), each an array of shape (n_components,).
        invert_y_axis:   Boolean to decide whether to invert the y-axis.
        save_path:       Optional file path to save the figure (e.g., "output.pdf").
    """
    seq_len = positions_array.shape[1]
    val_min, val_max = val_range
    global_min, global_max = global_extrema

    time_indices = np.arange(seq_len)
    num_labels = len(unique_labels)

    fig, axes = plt.subplots(
        n_components, num_labels,
        figsize=(5 * num_labels, 5 * n_components),
        squeeze=False
    )

    # List for storing one image object per component (for colorbar)
    im_list = []

    for comp_idx in range(n_components):
        comp_min = global_min[comp_idx]
        comp_max = global_max[comp_idx]

        for label_idx, label in enumerate(unique_labels):
            ax = axes[comp_idx, label_idx]

            # Plot raw trajectories in the background
            label_indices = np.where(sample_labels == label)[0]
            for i in label_indices:
                ax.plot(
                    time_indices,
                    positions_array[i, :, 0],
                    color='black',
                    alpha=0.01,
                    linewidth=0.5
                )

            # Get the stored heatmap
            heatmap_smoothed = heatmaps_dict.get((label, comp_idx), None)
            if heatmap_smoothed is None:
                continue

            # Plot heatmap with fixed global min/max
            im = ax.imshow(
                heatmap_smoothed.T,
                extent=[0, seq_len, val_min, val_max],
                origin='lower',
                cmap='YlOrRd',
                aspect='auto',
                vmin=comp_min,  # Use global minimum for this component
                vmax=comp_max   # Use global maximum for this component
            )
            im_list.append(im)

            # Hide x-ticks except for the second row
            if comp_idx != 1:
                ax.set_xticklabels([])
                ax.set_xticks([])

            # Hide y-ticks except for the first column
            if label_idx != 0:
                ax.set_yticklabels([])
                ax.set_yticks([])

            if invert_y_axis:
                ax.invert_yaxis()

            # Add column labels (only on the first row)
            if comp_idx == 0:
                ax.set_title(f"Label: {label}", pad=10)

            # Add row labels (only on the first column)
            if label_idx == 0:
                ax.set_ylabel(f"PC {comp_idx+1}", rotation=90, labelpad=0)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit labels properly

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()



def plot_all_trajectories_with_and_without_heatmaps(
    heatmaps_dict,
    positions_array,
    sample_labels,
    unique_labels,
    n_components,
    val_range,
    global_extrema,
    invert_y_axis=False,
    save_path=None  # Optional path to save the figure
):
    """
    1) Plots a single subplot with all time series (color-coded by label).
    2) For each component, plots the SUM of all label-based heatmaps on the
       same global brightness scale.

    Args:
        heatmaps_dict:  Dictionary keyed by (label, comp_idx) -> 2D heatmap.
        positions_array: Array of shape (num_samples, seq_len, 1).
        sample_labels:  Array of shape (num_samples,).
        unique_labels:  Unique labels.
        n_components:   Number of PCA components.
        val_range:      Tuple (val_min, val_max) across the entire dataset.
        global_extrema: Tuple (global_min, global_max), each an array of shape (n_components,).
        invert_y_axis:  Boolean to decide whether to invert the y-axis.
        save_path:      Optional file path to save the figure (e.g., "combined_output.pdf").
    """
    seq_len = positions_array.shape[1]
    val_min, val_max = val_range
    global_min, global_max = global_extrema

    num_samples = positions_array.shape[0]
    time_indices = np.arange(seq_len)

    # Define colors for labels (adjust as needed)
    color_map = {
        '3': '#4daf4a',
        '6': '#1f77b4',
        '7': '#e41a1c',
        3: '#4daf4a',
    }

    fig, axes = plt.subplots(1, n_components + 1, figsize=(6 * (n_components + 1), 6))

    # Left subplot: all time series color-coded by label
    ax_all = axes[0]
    for label in unique_labels:
        label_indices = np.where(sample_labels == label)[0]
        color = color_map.get(label, 'black')
        for idx in label_indices:
            ax_all.plot(
                time_indices,
                positions_array[idx, :, 0],
                color=color,
                alpha=0.2,
                linewidth=0.5
            )
    ax_all.set_title("All Time Series (by Label)")
    ax_all.set_xlabel("Time")
    ax_all.set_ylabel("Value")
    if invert_y_axis:
        ax_all.invert_yaxis()

    # For each component, create combined heatmap (sum of all labels)
    for comp_idx in range(n_components):
        ax = axes[comp_idx + 1]

        # Plot background time series in black
        for i in range(num_samples):
            ax.plot(
                time_indices,
                positions_array[i, :, 0],
                color='black',
                alpha=0.05,
                linewidth=0.3
            )

        # Sum across all labels
        heatmap_sum = None
        for label in unique_labels:
            hm_label = heatmaps_dict.get((label, comp_idx), None)
            if hm_label is not None:
                if heatmap_sum is None:
                    heatmap_sum = hm_label.copy()
                else:
                    heatmap_sum += hm_label

        if heatmap_sum is not None:
            # Normalize using the global scale for this component
            comp_min = global_min[comp_idx]
            comp_max = global_max[comp_idx]
            # The normalization here is not altering the values for imshow,
            # because we set the global scale through vmin/vmax if needed.
            ax.imshow(
                heatmap_sum.T,
                extent=[0, seq_len, val_min, val_max],
                origin='lower',
                cmap='YlOrRd',
                aspect='auto',
                vmin=comp_min,
                vmax=comp_max
            )

        ax.set_title(f"Combined Heatmap - PC {comp_idx+1}")
        ax.set_xlabel("Time")
        if comp_idx == 0:
            ax.set_ylabel("Value")
        else:
            ax.set_yticks([])
        if invert_y_axis:
            ax.invert_yaxis()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    
    plt.show()
    
def interactive_pca_heatmap_overlay_selection(
    scores,
    sample_labels,
    positions_array,
    gradcam_projections,
    n_components=2,
    grid_size=200,
    sigma=5,
    bin_stats='mean',
    normalize=True,
    invert_y_axis=False,
    color_map=None
):
    """
    Creates a Panel dashboard arranged in three columns:
      Left: a dataset plot (one axis per class, vertically arranged),
      Center: an interactive PCA scatter plot of latent vectors (square aspect),
      Right: heatmap overlays computed for selected samples.
    
    When points are selected on the PCA scatter plot, the heatmap pane updates.
    
    Parameters:
        scores (np.ndarray): PCA scores (n_samples, 2) from PCA on the latent vectors.
        sample_labels (np.ndarray): Array of sample labels (n_samples,).
        positions_array (np.ndarray): Time series data (n_samples, seq_len, 1).
        gradcam_projections (np.ndarray): Grad-CAM projections (n_samples, seq_len, n_components).
        n_components (int): Number of PCA components.
        grid_size (int): Number of bins along the value axis.
        sigma (float): Smoothing sigma for the heatmap.
        bin_stats (str): Statistic for binning (e.g. 'mean').
        normalize (bool): Whether to normalize the heatmaps.
        invert_y_axis (bool): Whether to invert the y-axis.
        color_map (dict): Optional mapping from labels to colors (used for trajectory plotting).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import panel as pn
    import plotly.graph_objects as go

    # Initialize Panel extension for Plotly.
    pn.extension('plotly')

    # Set matplotlib styling.
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    # Use a default color mapping if none provided.
    if color_map is None:
        color_map = {
            '3': '#4daf4a',
            '6': '#1f77b4',
            '7': '#e41a1c',
            3: '#4daf4a'
        }
    marker_colors = [color_map.get(lab, 'black') for lab in sample_labels]

    # Create an interactive Plotly scatter plot with square aspect.
    fig = go.FigureWidget(
        data=[go.Scatter(
            x=scores[:, 0],
            y=scores[:, 1],
            mode='markers',
            marker=dict(size=8, color=marker_colors),
            customdata=sample_labels,
            name='Latent Vectors'
        )],
        layout=go.Layout(
            title=dict(text="PCA Projection of Latent Vectors (Interactive - Brush to Select)",
                       font=dict(family="serif", size=14)),
            xaxis=dict(
                title=dict(text="PC 1", font=dict(family="serif", size=12)),
                scaleanchor="y",   # Force x-axis scale to match y-axis
                scaleratio=1,
                autorange='reversed'
            ),
            yaxis=dict(title=dict(text="PC 2", font=dict(family="serif", size=12))),
            dragmode='select'
        )
    )

    # Wrap the Plotly figure in a Panel pane.
    plotly_pane = pn.pane.Plotly(fig, sizing_mode="stretch_both", config={'scrollZoom': True})

    # Create an initial placeholder Matplotlib figure for the heatmaps.
    init_fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, "Select points to view heatmaps", 
            horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.close(init_fig)
    heatmap_pane = pn.pane.Matplotlib(init_fig, tight=True, sizing_mode="stretch_both")


    # Create a dataset plot: one axis per class (vertically arranged).
    unique_labels_dataset = np.unique(sample_labels)
    num_classes = len(unique_labels_dataset)
    fig_dataset, axes_dataset = plt.subplots(num_classes, 1, figsize=(8, 3*num_classes))
    # Ensure axes_dataset is iterable even for a single class.
    if num_classes == 1:
        axes_dataset = [axes_dataset]
    time_indices = np.arange(positions_array.shape[1])
    for idx, label in enumerate(unique_labels_dataset):
        ax = axes_dataset[idx]
        class_indices = np.where(sample_labels == label)[0]
        for i in class_indices:
            ax.plot(time_indices, positions_array[i, :, 0], 
                    color=color_map.get(label, 'black'), alpha=0.1)
        ax.set_title(f"Class: {label}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
    plt.tight_layout()
    dataset_pane = pn.pane.Matplotlib(fig_dataset, tight=True, sizing_mode="stretch_both")

    # Import the heatmap computation function.
    from explain.gradcam_heatmap_timeseries import compute_binned_heatmaps_per_label_and_component

    # Define the selection callback.
    def selection_callback(trace, points, selector):
        if len(points.point_inds) == 0:
            print("No points selected.")
            return

        selected_inds = points.point_inds
        sel_positions = positions_array[selected_inds]
        sel_gradcam = gradcam_projections[selected_inds]
        sel_labels = sample_labels[selected_inds]
        unique_sel_labels = np.unique(sel_labels)
        print(f"Selected {len(selected_inds)} points; labels: {unique_sel_labels}")

        # Compute heatmaps.
        heatmaps_dict, val_range, (global_min, global_max) = compute_binned_heatmaps_per_label_and_component(
            sel_positions,
            sel_gradcam,
            sel_labels,
            unique_sel_labels,
            n_components=n_components,
            grid_size=grid_size,
            bin_stats=bin_stats,
            sigma=sigma,
            normalize=normalize
        )

        seq_len_local = sel_positions.shape[1]
        time_indices = np.arange(seq_len_local)
        num_labels = len(unique_sel_labels)

        # Create a new Matplotlib figure for heatmap overlays.
        fig_overlay, axes = plt.subplots(
            n_components, num_labels,
            figsize=(5 * num_labels, 5 * n_components),
            squeeze=False
        )

        for comp_idx in range(n_components):
            comp_min = global_min[comp_idx]
            comp_max = global_max[comp_idx]
            for label_idx, label in enumerate(unique_sel_labels):
                ax = axes[comp_idx, label_idx]
                # Plot raw trajectories.
                label_inds = np.where(sel_labels == label)[0]
                for i in label_inds:
                    ax.plot(
                        time_indices,
                        sel_positions[i, :, 0],
                        color='black',
                        alpha=0.01,
                        linewidth=0.5
                    )
                # Overlay heatmap.
                heatmap = heatmaps_dict.get((label, comp_idx), None)
                if heatmap is not None:
                    ax.imshow(
                        heatmap.T,
                        extent=[0, seq_len_local, val_range[0], val_range[1]],
                        origin='lower',
                        cmap='YlOrRd',
                        aspect='auto',
                        vmin=comp_min,
                        vmax=comp_max
                    )
                # Format axes.
                if comp_idx != n_components - 1:
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                if label_idx != 0:
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                if comp_idx == 0:
                    ax.set_title(f"Label: {label}", pad=10)
                if label_idx == 0:
                    ax.set_ylabel(f"PC {comp_idx+1}", rotation=90, labelpad=5)
                if invert_y_axis:
                    ax.invert_yaxis()

        plt.tight_layout()
        # Update the heatmap pane.
        heatmap_pane.object = fig_overlay

    # Connect the selection callback.
    scatter = fig.data[0]
    scatter.on_selection(selection_callback)

    # Create a three-column layout:
    # Left: dataset plot.
    # Center: interactive PCA scatter plot (with square aspect).
    # Right: heatmap overlays.
    dashboard = pn.Row(dataset_pane, plotly_pane, heatmap_pane)
    # Start the Panel server and open the dashboard in a new browser window.
    pn.serve(dashboard, show=True)


# Example usage:
# interactive_pca_heatmap_overlay_selection(scores, sample_labels, positions_array, gradcam_projections)
