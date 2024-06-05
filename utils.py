import numpy as np
import torch


CONSTANT = 1e-5


def normalize_mels(x: torch.FloatTensor, sequence_lengths: torch.IntTensor):
    """
    Normalize a batch of mel spectrograms x with the shape (batch_size, n_mels, time)
    Source: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py
    """
    batch_size = x.shape[0]
    max_time = x.shape[2]

    time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
    valid_mask = time_steps < sequence_lengths.unsqueeze(1)
    x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
    x_mean_denominator = valid_mask.sum(axis=1)
    x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)

    # Subtract 1 in the denominator to correct for the bias.
    x_std = torch.sqrt(
        torch.sum(torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
        / (x_mean_denominator.unsqueeze(1) - 1.0)
    )

    # make sure x_std is not zero
    x_std += CONSTANT

    return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)


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
