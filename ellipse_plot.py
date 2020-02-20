import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim


# this script will just be used to plot the results from the libertem output

# we do not fit all the patterns in the vacuum, so we need to put them back into the correct shape for easy slicing
fit_stack = np.zeros((128, 65, 12))
fit_stack[roi_pillar] = fit

# we want background to be black so we slightly modify our chosen colormap such that nans are black
cmap = plt.cm.viridis
cmap.set_bad(color="k")

# we now make an appropriately sized array of nans
im = np.full((128, 65), np.nan)

# we filter our data with zeros on the edges (for vacuum) as filtering with nans does not work well, and then put it into the array of nans.
# this all uses the roi_pillar that is a binary mask for our navigation axes
im[roi_pillar] = spim.median_filter(fit_stack[:, :, 0], 8)[roi_pillar]

# then we plot, using our cmap with nans being shown by black
plt.figure(12, clear=True)
plt.imshow(im, cmap=cmap)
