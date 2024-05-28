import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from datasets import Dataset
from utils import proper_n_bins


# Special chars
CHAR_SPACE = ' '
CHAR_PAD = '[PAD]'
CHAR_UNK = '[UNK]'

# Dataset address
CACHE_DIR = 'cache'
DS_ROOT = 'dataset'
WAVS_DIR = f'{DS_ROOT}/wav'
DS_SR = 22050


def read_lines(file_path):
    return [l.strip() for l in open(file_path) if l.strip() != '']


def preprocess_text(s):
    """"
    Clean the labels

    Special chars kept in the dataset:
        - Chars with Hamza above: u0623, u0624, u0626
        - Arabic Chars: u0629 (ة), 
    """

    # Remove unwanted chars
    chars_to_remove = [
        u'\u064d', # kasratan
        u'\u064b', # fathatan
    ]
    for c in chars_to_remove:
        s = s.replace(c, ' ')

    # Replace chars with their more common form
    chars_to_replace = [
        (u'\u0625', 'ا'),
        (u'\u200c', ' '),
    ]
    for c_old, c_new in chars_to_replace:
        s = s.replace(c_old, c_new)

    s = s.replace('sil', ' ')
    s = re.sub('[\s\r\n]+', ' ', s).strip()
    return s


def load_split(split_name):
    """Load a subset of the dataset"""
    assert split_name in ['train', 'test'], "Invalid dataset split"

    segments_path = f'{DS_ROOT}/{split_name}/segments'
    slines = read_lines(segments_path)
    segments = {l[:14] : tuple(map(float, l[21:].split())) for l in slines}

    text_path = f'{DS_ROOT}/{split_name}/text'
    tlines = read_lines(text_path)
    labels = {l[:14] : l[15:] for l in tlines}

    data = {k: {
        'sentence': preprocess_text(labels[k]),
        'segment': segments[k],
        'wav_path': os.path.join(WAVS_DIR, f'{k[:5]}.wav')
    } for k in labels.keys()}
    return data


def plot_duration_histograms():
    """Plot the duration histogram of the dataset"""
    # Find train and test durations
    train_data = list(load_split('train').values())
    test_data = list(load_split('test').values())
    train_durations = np.array([item['segment'][1] - item['segment'][0] for item in train_data])
    test_durations = np.array([item['segment'][1] - item['segment'][0] for item in test_data])

    print('Share of items shorter than 60 seconds:')
    print(f'Train: {(train_durations < 60).sum() / len(train_durations)}')  # 0.9927385892116183
    print(f'Test: {(test_durations < 60).sum() / len(test_durations)}')     # 0.9847953216374269

    # Plot histograms
    train_bins = proper_n_bins(train_durations)
    test_bins = proper_n_bins(test_durations)
    plt.hist(train_durations, density=True, bins=train_bins, color='blue', alpha=0.4)
    plt.hist(test_durations, density=True, bins=test_bins, color='red', alpha=0.4)
    plt.legend(['Train', 'Test'])
    plt.xlabel('Duration (second)')
    plt.ylabel('Density (%)')
    plt.title('Audio Duration Histogram')
    plt.savefig('DurationHistogram.png')


def generate_vocab(with_special_tokens=True):
    train_data = load_split('train')
    test_data = load_split('test')
    all_data = train_data | test_data
    labels = [item['sentence'] for item in all_data.values()]
    all_chars = list(set(' '.join(labels)))
    all_chars = sorted(all_chars, key=ord)
    vocab_dict = {v: k for k, v in enumerate(all_chars)}

    # Space
    if CHAR_SPACE != ' ':
        vocab_dict[CHAR_SPACE] = vocab_dict[" "]
        del vocab_dict[" "]

    # Special tokens
    if with_special_tokens:
        vocab_dict[CHAR_UNK] = len(vocab_dict)
        vocab_dict[CHAR_PAD] = len(vocab_dict)

    # Save the vocab
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


def load_item(item, sampling_rate):
    """Load and resample a wav file"""
    waveform, orig_sr = torchaudio.load(item['wav_path'])

    # Extract the audio segment
    start_second, end_second = item['segment']
    start_sample, end_sample = round(orig_sr * start_second), round(orig_sr * end_second)
    waveform = waveform[:, start_sample:end_sample]

    # Resample if required
    if orig_sr != sampling_rate:
        waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=sampling_rate)    

    return {'waveform': waveform, 'sampling_rate': sampling_rate}


def load_hf_dataset(split_name, sampling_rate=DS_SR):
    """Load the dataset as a Hugging Face dataset"""
    assert split_name in ['train', 'test'], "Invalid dataset split"
    data = list(load_split(split_name).values())
    dataset = Dataset.from_list(data)
    dataset = dataset.map(
        lambda item: load_item(item, sampling_rate),
        remove_columns=['segment', 'wav_path'],
        keep_in_memory=False,
        cache_file_name=os.path.join(CACHE_DIR, split_name)
    )
    dataset = dataset.with_format("torch")
    return dataset


if __name__ == '__main__':
    generate_vocab()
    plot_duration_histograms()
