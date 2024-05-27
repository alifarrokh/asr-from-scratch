import os
import re
import json


# Special chars
CHAR_SPACE = '|'
CHAR_PAD = '[PAD]'
CHAR_UNK = '[UNK]'

# Dataset address
DS_ROOT = 'dataset'
WAVS_DIR = f'{DS_ROOT}/wav'


def read_lines(file_path):
    return [l.strip() for l in open(file_path) if l.strip() != '']


def preprocess_text(s):
    """"
    Chars with Hamza above: u0623, u0624, u0626
    Arabic Chars: u0629 (ة), 
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
    assert split_name in ['train', 'test'], "Invalid dataset split"

    segments_path = f'{DS_ROOT}/{split_name}/segments'
    slines = read_lines(segments_path)
    segments = {l[:14] : tuple(map(float, l[21:].split())) for l in slines}

    text_path = f'{DS_ROOT}/{split_name}/text'
    tlines = read_lines(text_path)
    labels = {l[:14] : l[15:] for l in tlines}

    data = {k: {
        'label': preprocess_text(labels[k]),
        'segment': segments[k],
        'wav_path': os.path.join(WAVS_DIR, f'{k[:5]}.wav')
    } for k in labels.keys()}
    return data


def generate_vocab(with_special_tokens=True):
    train_data = load_split('train')
    test_data = load_split('test')
    all_data = train_data | test_data
    labels = [item['label'] for item in all_data.values()]
    all_chars = list(set(' '.join(labels)))
    all_chars = sorted(all_chars, key=ord)
    vocab_dict = {v: k for k, v in enumerate(all_chars)}

    # Space
    vocab_dict[CHAR_SPACE] = vocab_dict[" "]
    del vocab_dict[" "]

    # Special tokens
    if with_special_tokens:
        vocab_dict[CHAR_UNK] = len(vocab_dict)
        vocab_dict[CHAR_PAD] = len(vocab_dict)

    # Save the vocab
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


if __name__ == '__main__':
    generate_vocab()
