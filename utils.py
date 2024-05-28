import numpy as np


def proper_n_bins(x):
    """Find the proper number of bins for plotting the histogram of x"""
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    bins = round((x.max() - x.min()) / bin_width)
    return bins
