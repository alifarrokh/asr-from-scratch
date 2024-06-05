import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from datasets import Dataset
from utils import proper_n_bins


# Dataset Config
CACHE_DIR = 'cache'
DS_ROOT = 'dataset'
WAVS_DIR = f'{DS_ROOT}/wav'
DS_SR = 22050

# Create cache directory
os.system(f'mkdir -p {CACHE_DIR}')


def read_lines(file_path):
    """Return the list of lines in the given file"""
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


def load_split(split_name, max_duration: int = 22):
    """Load a dataset subset"""
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

    # Filter long audio segments
    if max_duration:
        duration = lambda item: item['segment'][1] - item['segment'][0]
        data = {k:v for k,v in data.items() if duration(v) < max_duration}

    return data


def plot_duration_histograms(max_duration: int):
    """Plot the duration histogram of the dataset"""
    # Find train and test durations
    train_data = list(load_split('train', max_duration=10000).values())
    test_data = list(load_split('test', max_duration=10000).values())
    train_durations = np.array([item['segment'][1] - item['segment'][0] for item in train_data])
    test_durations = np.array([item['segment'][1] - item['segment'][0] for item in test_data])

    print(f'Share of items shorter than {max_duration} seconds:')
    print(f'Train: {(train_durations < max_duration).sum() / len(train_durations)}')
    print(f'Test: {(test_durations < max_duration).sum() / len(test_durations)}')

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


def get_alphabet():
    """Extract the vocabulary from the dataset"""
    train_data = load_split('train')
    test_data = load_split('test')
    all_data = train_data | test_data
    labels = [item['sentence'] for item in all_data.values()]
    all_chars = list(set(' '.join(labels)))
    all_chars = sorted(all_chars, key=ord)
    return all_chars


def load_item(item):
    """Load and resample an item"""
    sampling_rate = item['target_sampling_rate']
    waveform, orig_sr = torchaudio.load(item['wav_path'])

    # Extract the audio segment
    start_second, end_second = item['segment']
    start_sample, end_sample = round(orig_sr * start_second), round(orig_sr * end_second)
    waveform = waveform[:, start_sample:end_sample]

    # Resample if required
    if orig_sr != sampling_rate:
        waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=sampling_rate)    

    return {'waveform': waveform, 'sampling_rate': sampling_rate}


def extract_features(item):
    """Extract Mel features"""
    mel_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate=item['sampling_rate'],
        n_fft=400, # 25 ms for 16K hz audio
        hop_length=160, # 10 ms stride
        f_min=0,
        f_max=item['sampling_rate'] // 2,
        n_mels=80
    )
    return {'mel_spectrogram': mel_extractor(item['waveform'])}


def load_hf_dataset(
    split_name: str,
    sampling_rate: int = DS_SR,
    with_features: bool = False,
    max_duration: int = 22,
    limit: int = None, # Max number of returned items (for testing purposes)
) -> Dataset:
    """Load the dataset as a Hugging Face dataset"""
    assert split_name in ['train', 'test'], "Invalid dataset split"

    # Load metadata
    data = list(load_split(split_name, max_duration).values())
    data = [item | {'target_sampling_rate': sampling_rate} for item in data]
    if limit:
        data = data[:limit]

    # Create dataset
    limit_str = f'_{limit}' if limit else ''
    dataset = Dataset.from_list(data)
    dataset = dataset.map(
        load_item,
        remove_columns=['segment', 'wav_path', 'target_sampling_rate'],
        keep_in_memory=False,
        cache_file_name=os.path.join(CACHE_DIR, f'{split_name}{limit_str}_raw_{max_duration}s')
    )
    dataset = dataset.with_format("torch")

    # Extract features
    if with_features:
        dataset = dataset.map(
            extract_features,
            remove_columns=['sampling_rate', 'waveform'],
        )

    return dataset


def extract_train_sentences():
    """
    Save the list of training sentences (required for training a language model)
    """
    train_data = list(load_split('train', max_duration=1000).values())
    with open('train_sentences.txt', 'w') as out:
        for item in train_data:
            out.write(item['sentence'] + '\n')


if __name__ == '__main__':
    plot_duration_histograms(max_duration=22)
    extract_train_sentences()
