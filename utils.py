import numpy as np


def proper_n_bins(x):
    """Find the proper number of bins for plotting the histogram of x"""
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    bins = round((x.max() - x.min()) / bin_width)
    return bins


def describe_model_size(model_instance, with_layers=True):
    """Describes the given model's size"""
    total_params = 0
    total_mem = 0
    if with_layers:
        print('===================== Layers =====================')
    for name, param in model_instance.named_parameters():
        if with_layers:
            print(f"{param.nelement():<12,} {name}")
        total_params += param.nelement()
        total_mem += param.element_size() * param.nelement()
    if with_layers:
        print()
    print(f'Total Params: {total_params:,}')
    print(f'Size in memory: {total_mem/(2**20):.3f} MB')
