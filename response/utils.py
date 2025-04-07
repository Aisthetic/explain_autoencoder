from ast import List
import numpy as np
from sympy import Float, plot
import torch
from torch import Tensor, nn
from deep_traffic_generation.core.abstract import Abstract
from deep_traffic_generation.core.utils import init_dataframe
import pandas as pd
# from mpl_toolkits.basemap import Basemap

def make_MLP(din, dout, hidden=None, nonlin=nn.ELU, output_nonlin=None, bias=True, output_bias=None):
	'''
	:param din: int
	:param dout: int
	:param hidden: ordered list of int - each element corresponds to a FC layer with that width (empty means network is not deep)
	:param nonlin: str - choose from options found in get_nonlinearity(), applied after each intermediate layer
	:param output_nonlin: str - nonlinearity to be applied after the last (output) layer
	:return: an nn.Sequential instance with the corresponding layers
	'''

	if hidden is None:
		hidden = []

	if output_bias is None:
		output_bias = bias

	flatten = False
	reshape = None

	if isinstance(din, (tuple, list)):
		flatten = True
		din = int(np.product(din))
	if isinstance(dout, (tuple, list)):
		reshape = dout
		dout = int(np.product(dout))

	nonlins = [nonlin] * len(hidden) + [output_nonlin]
	biases = [bias] * len(hidden) + [output_bias]
	hidden = din, *hidden, dout

	layers = []
	if flatten:
		layers.append(nn.Flatten())

	for in_dim, out_dim, nonlin, bias in zip(hidden, hidden[1:], nonlins, biases):
		layer = nn.Linear(in_dim, out_dim, bias=bias)
		layers.append(layer)
		if nonlin is not None:
			layers.append(nonlin())

	if reshape is not None:
		layers.append(Reshaper(reshape))


	net = nn.Sequential(*layers)

	net.din, net.dout = din, dout
	return net



class Reshaper(nn.Module): # by default flattens
	def __init__(self, dout=(-1,)):
		super().__init__()

		self.dout = dout


	def extra_repr(self):
		return f'out={self.dout}'


	def forward(self, x):
		B = x.size(0)
		return x.view(B, *self.dout)



import matplotlib.pyplot as plt
from matplotlib.figure import figaspect



def factors(n): # has duplicates, starts from the extremes and ends with the middle
	return (x for tup in ([i, n//i]
				for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup)



def calc_tiling(N, H=None, W=None, prefer_tall=False):

	if H is not None and W is None:
		W = N//H
	if W is not None and H is None:
		H = N//W

	if H is not None and W is not None and N == H*W:
		return H, W

	H,W = tuple(factors(N))[-2:] # most middle 2 factors

	if H > W or prefer_tall:
		H, W = W, H

	return H, W



def plot_imgs(imgs, H=None, W=None,
              figsize=None, scale=1,
              reverse_rows=False, grdlines=False,
              channel_first=None,
              imgroot=None, params={},
              savepath=None, dpi=96, autoclose=True, savescale=1,
              adjust={}, border=0., between=0.01):
    """
    Plot a grid of images.

    Args:
        imgs (torch.Tensor or list or tuple): Images to plot. Can be a torch.Tensor,
            list, or tuple containing torch.Tensor or numpy arrays.
        H (int): Number of rows in the grid. If not specified, it will be calculated automatically.
        W (int): Number of columns in the grid. If not specified, it will be calculated automatically.
        figsize (tuple): Figure size (width, height) in inches. If not specified, it will be calculated automatically.
        scale (float): Scale factor for the figure size.
        reverse_rows (bool): Flag to reverse the order of rows in the grid.
        grdlines (bool): Flag to draw gridlines on the images.
        channel_first (bool): Flag to indicate if the channels are in the first dimension of the images.
            If not specified, it will be determined automatically.
        imgroot (str): Root directory for the images.
        params (dict): Additional parameters to pass to the plt.imshow function.
        savepath (str): File path to save the figure. If not specified, the figure will not be saved.
        dpi (int): Dots per inch for saving the figure.
        autoclose (bool): Flag to automatically close the figure after saving.
        savescale (float): Scale factor for the saved figure size.
        adjust (dict): Additional parameters to adjust the spacing and borders of the subplots.
        border (float): Border width as a fraction of the figure size.
        between (float): Spacing between subplots as a fraction of the figure size.

    Returns:
        fg (matplotlib.figure.Figure): The created figure object.
        axes (numpy.ndarray): Array of axes objects representing the subplots.
    """

    # Convert torch.Tensor to numpy array if necessary
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().squeeze(0).numpy()
    # Convert list of torch.Tensor to list of numpy arrays if necessary
    elif isinstance(imgs, (list, tuple)):
        imgs = [img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img for img in imgs]

    if isinstance(imgs, np.ndarray):
        shape = imgs.shape

        # Check if the channels are in the first dimension or the last dimension
        if channel_first is None and shape[0 if len(shape) == 3 else 1] in {1, 3, 4} and shape[-1] not in {1, 3, 4}:
            channel_first = True

        # Reshape the array based on its dimensions
        if len(shape) == 2 or (len(shape) == 3 and ((shape[0] in {1, 3, 4} and channel_first)
                                                    or (shape[-1] in {1, 3, 4} and not channel_first))):
            imgs = [imgs]
        elif len(shape) == 4:
            if channel_first:
                # Transpose the dimensions to have channels as the last dimension
                imgs = imgs.transpose(0, 2, 3, 1)
                channel_first = False
        else:
            raise Exception(f'unknown shape: {shape}')

    # Reshape the images to have the channel dimension last
    imgs = [img.transpose(1, 2, 0).squeeze() if channel_first and len(img.shape) == 3 else img.squeeze() for img in
            imgs]

    iH, iW = imgs[0].shape[:2]

    # Calculate the number of rows (H) and columns (W) for tiling the images
    H, W = calc_tiling(len(imgs), H=H, W=W)

    fH, fW = iH * H, iW * W

    aw = None
    if figsize is None:
        # Calculate the aspect ratio of the figure based on the image dimensions
        aw, ah = figaspect(fH / fW)
        aw, ah = scale * aw, scale * ah
        figsize = aw, ah

    # Create the figure and axes for plotting
    fg, axes = plt.subplots(H, W, figsize=figsize)
    axes = [axes] if len(imgs) == 1 else list(axes.flat)

    # Plot the images on the axes
    for ax, img in zip(axes, imgs):
        plt.sca(ax)
        if reverse_rows:
            # Reverse the order of rows in the image
            img = img[::-1]
        plt.imshow(img, **params)
        if grdlines:
            # Draw gridlines on the image
            plt.plot([0, iW], [iH / 2, iH / 2], c='r', lw=.5, ls='--')
            plt.plot([iW / 2, iW / 2], [0, iH], c='r', lw=.5, ls='--')
            plt.xlim(0, iW)
            plt.ylim(0, iH)

        plt.axis('off')

    if adjust is not None:
        # Adjust the spacing and borders of the subplots
        base = dict(wspace=between, hspace=between,
                    left=border, right=1 - border, bottom=border, top=1 - border)
        base.update(adjust)
        plt.subplots_adjust(**base)

    if savepath is not None:
        # Save the figure to a file
        plt.savefig(savepath, dpi=savescale * (dpi if aw is None else fW / aw))
        if autoclose:
            plt.close()
            return

    return fg, axes


def plot_mat(M, val_fmt=None, figax=None, figsize=None, figside=0.7,
             edgeeps=0.03, text_kwargs=dict(), dpi=300, **kwargs):
	if figax is None:
		figax = plt.subplots(figsize=figsize, dpi=dpi)
	fg, ax = figax

	plt.sca(ax)
	if isinstance(M, torch.Tensor):
		M = M.cpu().detach().numpy()
	if len(M.shape) == 1:
		M = M.reshape(1, -1)
	H, W = M.shape
	if figsize is None:
		fg.set_size_inches(figside * W + 0.5, figside * H + 0.5)
	plt.matshow(M, False, **kwargs)
	plt.yticks(np.arange(H), np.arange(H))
	plt.xticks(np.arange(W), np.arange(W))
	plt.subplots_adjust(edgeeps, edgeeps, 1 - edgeeps, 1 - edgeeps)
	if val_fmt is not None:
		if isinstance(val_fmt, int):
			val_fmt = f'.{val_fmt}g'
		if isinstance(val_fmt, str):
			val_fmt = '{:' + val_fmt + '}'
			fmt = lambda x: val_fmt.format(x)
		else:
			fmt = val_fmt

		if 'va' not in text_kwargs:
			text_kwargs['va'] = 'center'
		if 'ha' not in text_kwargs:
			text_kwargs['ha'] = 'center'
		for (i, j), z in np.ndenumerate(M):
			ax.text(j, i, fmt(z), **text_kwargs)
	return fg, ax

def plot_mat_latex(M, val_fmt=None, figax=None, figsize=None, figside=0.7,
             edgeeps=0.03, text_kwargs=dict(), dpi=300, save_path=None, **kwargs):
    """
    Parameters description here...
    """
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    
    if figax is None:
        figax = plt.subplots(figsize=figsize, dpi=dpi)
    fg, ax = figax

    plt.sca(ax)
    if isinstance(M, torch.Tensor):
        M = M.cpu().detach().numpy()
    if len(M.shape) == 1:
        M = M.reshape(1, -1)
    H, W = M.shape
    
    if figsize is None:
        fg.set_size_inches(figside * W + 0.5, figside * H + 0.5)
    
    plt.matshow(M, fignum=False, **kwargs)
    plt.yticks(np.arange(H), np.arange(H))
    plt.xticks(np.arange(W), np.arange(W))
    plt.subplots_adjust(edgeeps, edgeeps, 1 - edgeeps, 1 - edgeeps)
    
    if val_fmt is not None:
        if isinstance(val_fmt, int):
            val_fmt = f'.{val_fmt}g'
        if isinstance(val_fmt, str):
            val_fmt = '{:' + val_fmt + '}'
            fmt = lambda x: val_fmt.format(x)
        else:
            fmt = val_fmt

        if 'va' not in text_kwargs:
            text_kwargs['va'] = 'center'
        if 'ha' not in text_kwargs:
            text_kwargs['ha'] = 'center'
        
        for (i, j), z in np.ndenumerate(M):
            ax.text(j, i, fmt(z), **text_kwargs)
    
    if save_path is not None:
        plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')
    
    return fg, ax

def build_trajectories(trajectories, model):
	# Trajectories shape : (nb_samples, seq_len, nb_features)

	# detach the data from the graph$
	data = trajectories.view((trajectories.shape[0], -1)).detach().cpu().numpy()
	# unscale the data
	if model.dataset_params["scaler"] is not None:
		data = model.dataset_params["scaler"].inverse_transform(data)

	if isinstance(data, torch.Tensor):
		data = data.numpy()
	# get builder
	builder = model.get_builder(
		nb_samples=trajectories.shape[0], length=model.dataset_params["seq_len"]
	)
	# get features, track_unwrapped -> track
	features = [
		"track" if "track" in f else f for f in model.hparams.features 
	]
	# reshape data to (nb_samples * seq_len, nb_features)
	data = data.reshape((-1, len(features)))

	# Airport coordinates
	coordinates = dict(latitude=47.546585, longitude=8.447731)

	# add longitude and latitude to data with NaN values
	data = np.concatenate(
		[data, np.full((data.shape[0], 2), np.nan)], axis=1
	)

	# set the last point of each trajectory to the airport coordinates
	data[:: model.dataset_params["seq_len"], -2:] = [
		coordinates["longitude"], coordinates["latitude"]
	]

	# build dataframe from data and features
	df = pd.DataFrame(data, columns=[*features, "longitude", "latitude"])

	df = builder(df)

	# reverse the trajectories
	# df = df.iloc[::-1]

	return df

def plot_trajectories(trajectories, H=None, W=None,
			figsize=None, scale=1,
			reverse_rows=False, grdlines=False,
			channel_first=None,
			imgroot=None, params={},
			savepath=None, dpi=96, autoclose=True, savescale=1,
			adjust={}, border=0., between=0.01, mark_ends=False, on_map=False):
	"""
	Plot a grid of images.

	Args:
		trajectories (torch.Tensor): Trajectories to plot.
		H (int): Number of rows in the grid. If not specified, it will be calculated automatically.
		W (int): Number of columns in the grid. If not specified, it will be calculated automatically.
		figsize (tuple): Figure size (width, height) in inches. If not specified, it will be calculated automatically.
		scale (float): Scale factor for the figure size.
		reverse_rows (bool): Flag to reverse the order of rows in the grid.
		grdlines (bool): Flag to draw gridlines on the images.
		channel_first (bool): Flag to indicate if the channels are in the first dimension of the images.
			If not specified, it will be determined automatically.
		imgroot (str): Root directory for the images.
		params (dict): Additional parameters to pass to the plt.imshow function.
		savepath (str): File path to save the figure. If not specified, the figure will not be saved.
		dpi (int): Dots per inch for saving the figure.
		autoclose (bool): Flag to automatically close the figure after saving.
		savescale (float): Scale factor for the saved figure size.
		adjust (dict): Additional parameters to adjust the spacing and borders of the subplots.
		border (float): Border width as a fraction of the figure size.
		between (float): Spacing between subplots as a fraction of the figure size.

	Returns:
		fg (matplotlib.figure.Figure): The created figure object.
		axes (numpy.ndarray): Array of axes objects representing the subplots.
	"""

	# Convert the images to numpy arrays
	if isinstance(trajectories, torch.Tensor):
		trajectories = trajectories.detach().cpu().numpy()

	# Width and height of the trajectories
	iH, iW = trajectories[0].shape[1], trajectories[0].shape[1]

	# Calculate the number of rows (H) and columns (W) for tiling the images
	H, W = calc_tiling(len(trajectories), H=H, W=W)

	fH, fW = iH * H, iW * W

	aw = None
	if figsize is None:
		# Calculate the aspect ratio of the figure based on the image dimensions
		aw, ah = figaspect(fH / fW)
		aw, ah = scale * aw, scale * ah
		figsize = aw, ah

	# Create the figure and axes for plotting
	fg, axes = plt.subplots(H, W, figsize=figsize, dpi=300)

	# Ensure axes is always a 2D array
	if len(trajectories) == 1:
		axes = np.array([[axes]])
	else:
		axes = np.array(axes)

	# Force the same range of values for the axes using max and min values from the trajectories
	xmin = np.min([np.min(traj[:, 0]) for traj in trajectories])
	xmax = np.max([np.max(traj[:, 0]) for traj in trajectories])
	ymin = np.min([np.min(traj[:, 1]) for traj in trajectories])
	ymax = np.max([np.max(traj[:, 1]) for traj in trajectories])

	# Plot the images on the axes, loop in reverse
	# Swap the axes to plot the trajectories from top to bottom
	for ax, traj in zip(axes.flat, trajectories):
		plt.sca(ax)
		ax.set_xlim(xmin, xmax)
		ax.set_ylim(ymin, ymax)
		
		# Reverse the trajectory to start from the right corner
		traj = np.flip(traj, axis=0)

		# Plot the trajectory based on the longitude and latitude
		# traj is a 2D array of shape (H, W)
		if not on_map:
			plt.plot(traj[:, 0], traj[:, 1], **params)
			# add a red dot at the end of the trajectory
			if mark_ends:
				plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=2)
		else:
			plot_on_map(ax, traj)

		plt.axis('off')

	return fg, axes