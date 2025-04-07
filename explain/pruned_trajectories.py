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
    # positions.shape: (num_samples, seq_len, num_features)
    positions = np.swapaxes(positions, 1, 2)
    # After swapaxes, positions.shape: (num_samples, num_features, seq_len)
    print(f"Positions shape: {positions.shape}")

    labels = np.load(labels_path, allow_pickle=True)
    # labels.shape: (num_samples,)
    print(f"Labels shape: {labels.shape}")

    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")

    X = positions  # Shape: (num_samples, num_features, seq_len)
    y = None
    dataset = TSDataset(X, y)
    # dataset: custom dataset object

    model = CVAE.load_from_checkpoint(model_checkpoint_path, map_location=device, dataset_params=dataset.parameters).eval().to(device)
    # model: loaded CVAE model

    gcam = GradCAM(model, target_layer=target_layer, image_size=model.dataset_params["seq_len"], device=device)
    # gcam: initialized Grad-CAM object

    return dataset, labels, unique_labels, model, gcam

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

    # Now normalize over all samples for each latent dimension separately
    min_values = gradcam_activations.min(axis=(0, 1), keepdims=True)  # min over samples and sequence positions
    max_values = gradcam_activations.max(axis=(0, 1), keepdims=True)  # max over samples and sequence positions

    gradcam_activations_normalized = (gradcam_activations - min_values) / (max_values - min_values + 1e-8)

    sample_labels = np.array(sample_labels)
    # sample_labels.shape: (num_samples,)
    mu_list = np.array(mu_list)
    # mu_list.shape: (num_samples, latent_dim)

    return gradcam_activations_normalized, sample_labels, mu_list, positions_array

def perform_pca_on_latent_vectors(mu_list, n_components=2):
    """
    Performs PCA on the latent vectors and returns the PCA object.

    Args:
        mu_list: Numpy array of latent vectors, shape (num_samples, latent_dim)
        n_components: Number of principal components to keep.

    Returns:
        pca: Trained PCA object.
        scores: Transformed data, shape (num_samples, n_components)
    """
    print(f"Performing PCA on latent vectors with {n_components} components...")
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(mu_list)
    # scores.shape: (num_samples, n_components)
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Explained variance by each component: {pca.explained_variance_}")
    print(f"Components shape: {pca.components_.shape}")  # shape: (n_components, latent_dim)
    return pca, scores

def compute_projected_gradcam_activations(gradcam_activations, pca):
    """
    Computes the projected Grad-CAM activations onto PCA components.

    Args:
        gradcam_activations: Numpy array of Grad-CAM activations, shape (num_samples, seq_len, num_latent_layers)
        pca: Trained PCA object.

    Returns:
        gradcam_projections: Numpy array, shape (num_samples, seq_len, n_components)
    """
    num_samples, seq_len, num_latent_layers = gradcam_activations.shape
    print(f"Grad-CAM activations shape before projection: {gradcam_activations.shape}")

    # Reshape to (num_samples * seq_len, num_latent_layers)
    gradcam_activations_flat = gradcam_activations.reshape(-1, num_latent_layers)

    # Project onto PCA components
    gradcam_projections_flat = gradcam_activations_flat @ pca.components_.T  # Shape: (num_samples * seq_len, n_components)

    # Reshape back to (num_samples, seq_len, n_components)
    gradcam_projections = gradcam_projections_flat.reshape(num_samples, seq_len, -1)

    print(f"Grad-CAM projections shape: {gradcam_projections.shape}")
    return gradcam_projections

def compute_gradcam_with_pca_projection(
    positions_path, labels_path, model_checkpoint_path, target_layer, device='cpu', n_components=2
):
    """
    Computes Grad-CAM activations, performs PCA on latent vectors, and projects the Grad-CAM activations
    onto the first PCA component.

    Args:
        positions_path (str): Path to the positions .npy file.
        labels_path (str): Path to the labels .npy file.
        model_checkpoint_path (str): Path to the model checkpoint .ckpt file.
        target_layer (str): Name of the target layer in the model for Grad-CAM.
        device (str): Device to use ('cpu' or 'cuda').
        n_components (int): Number of PCA components to use.

    Returns:
        gradcam_first_pc: Projected Grad-CAM activations on the first PCA component, shape (num_samples, seq_len).
        positions_array: Original positions array, shape (num_samples, seq_len, num_features).
        sample_labels: Labels for each sample.
    """
    # Step 1: Load data and model
    dataset, labels, unique_labels, model, gcam = load_data_and_model(
        positions_path, labels_path, model_checkpoint_path, target_layer, device)
    seq_len = dataset[0][0].shape[1]
    print(f"Sequence length (seq_len): {seq_len}")

    # Step 2: Compute Grad-CAM activations and latent vectors
    gradcam_activations, sample_labels, mu_list, positions_array = compute_gradcam_per_sample(
        dataset, labels, model, gcam, device)

    # Step 3: Perform PCA on latent vectors
    pca, scores = perform_pca_on_latent_vectors(mu_list, n_components=n_components)

    # Step 4: Compute projected Grad-CAM activations onto PCA components
    gradcam_projections = np.abs(compute_projected_gradcam_activations(gradcam_activations, pca))

    # Get the projected Grad-CAM activations for the first PCA component
    gradcam_first_pc = gradcam_projections[:, :, 0]  # Shape: (num_samples, seq_len)

    return gradcam_first_pc, positions_array, sample_labels


import numpy as np

import numpy as np

def apply_pruning_mask(gradcam_first_pc, positions_array, sample_labels, threshold):
    """
    Applies a thresholding mask based on the Grad-CAM activations projected onto the first PCA component,
    and prunes positions based on this mask.

    Args:
        gradcam_first_pc (numpy.ndarray): Projected Grad-CAM activations on the first PCA component, shape (num_samples, seq_len).
        positions_array (numpy.ndarray): Original positions array, shape (num_samples, seq_len, num_features).
        sample_labels (list): Labels for each sample.
        threshold (float): Threshold for Grad-CAM activation.

    Returns:
        pruned_positions_array (numpy.ndarray): Array of pruned positions per sample, shape (num_samples, seq_len, num_features).
        pruned_labels (numpy.ndarray): Array of labels per sample, with '_pruned' appended.
        pruned_part (numpy.ndarray): Inverse of purned_positions_array, Array of the pruned positions per sample, shape (num_samples, seq_len, num_features).
    """
    # Create a mask based on the threshold as a percentage of the maximum value
    threshold_value = threshold * gradcam_first_pc.max()
    mask = gradcam_first_pc >= threshold_value  # Shape: (num_samples, seq_len)

    # Expand the mask to match the number of features
    mask_expanded = np.expand_dims(mask, axis=-1)  # Shape: (num_samples, seq_len, 1)
    mask_expanded = np.repeat(mask_expanded, positions_array.shape[2], axis=-1)  # Shape: (num_samples, seq_len, num_features)

    # Apply the mask to get the pruned positions
    pruned_positions_array = positions_array * mask_expanded  # Shape: (num_samples, seq_len, num_features)

    # Create the pruned part by capturing the positions where the mask is not applied (i.e., pruned positions)
    pruned_part = positions_array * (~mask_expanded)  # Shape: (num_samples, seq_len, num_features)

    # Calculate the percentage of data set to zero
    total_elements = np.prod(pruned_positions_array.shape)
    zero_elements = np.sum(pruned_positions_array == 0)
    zero_percentage = (zero_elements / total_elements) * 100
    print(f"Percentage of data set to zero: {zero_percentage:.2f}%")

    # Generate pruned labels by appending '_pruned'
    pruned_labels = np.array([str(label) + "_pruned" for label in sample_labels])

    return pruned_positions_array, pruned_labels, pruned_part
