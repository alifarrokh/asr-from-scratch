from dataclasses import dataclass
from typing import Union
import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class MelSpectrogramDataCollator:
    """ Mel Spectrogram Data Collator """

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        # Sort items based on sequence length
        items = sorted(items, key=lambda item: item['mel_spectrogram'].shape[-1], reverse=True)

        # Pad features
        mel_spectrograms = [item['mel_spectrogram'].squeeze(0).T for item in items]
        features = pad_sequence(mel_spectrograms, batch_first=True) # (batch_size, time, n_mels)
        features = features.transpose(1, 2).unsqueeze(1) # (batch_size, 1, n_mels, time)

        # Pad labels
        labels = [item['labels'] for item in items]
        labels = pad_sequence(labels, batch_first=True)

        # Compute senquence & label lengths
        sequence_lengths = [item['mel_spectrogram'].shape[-1] for item in items]
        label_lengths = [len(item['labels']) for item in items]

        batch = {
            'features': features,
            'labels': labels,
            'sequence_lengths': torch.tensor(sequence_lengths, dtype=torch.int),
            'label_lengths': torch.tensor(label_lengths, dtype=torch.int)
        }

        return batch
