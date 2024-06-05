from dataclasses import dataclass
from math import ceil
import torch
from torch.nn.utils.rnn import pad_sequence
from utils import normalize_mels


@dataclass
class MelSpectrogramDataCollator:
    """Mel Spectrogram Data Collator"""

    pad_mels_to_multiple_of: int = None
    normalize: bool = True

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        # Sort items based on sequence length
        items = sorted(items, key=lambda item: item['mel_spectrogram'].shape[-1], reverse=True)

        # Pad features
        mel_spectrograms = [item['mel_spectrogram'].squeeze(0) for item in items]

        if self.pad_mels_to_multiple_of:
            m = self.pad_mels_to_multiple_of
            first_larger_multiple = m * ceil(mel_spectrograms[0].shape[1] / m)
            diff = first_larger_multiple - mel_spectrograms[0].shape[1]
            pad = torch.zeros(mel_spectrograms[0].shape[0], diff, dtype=mel_spectrograms[0].dtype)
            mel_spectrograms[0] = torch.cat([mel_spectrograms[0], pad], dim=-1)

        mel_spectrograms = [mel.T for mel in mel_spectrograms]
        features = pad_sequence(mel_spectrograms, batch_first=True) # (batch_size, time, n_mels)
        features = features.transpose(1, 2) # (batch_size, n_mels, time)

        # Pad labels
        labels = [item['labels'] for item in items]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100) # (batch_size, time)

        # Compute senquence & label lengths
        sequence_lengths = [item['mel_spectrogram'].shape[-1] for item in items]
        label_lengths = [len(item['labels']) for item in items]

        # Normalize the features
        if self.normalize:
            features = normalize_mels(features, torch.tensor(sequence_lengths))

        batch = {
            'features': features,
            'labels': labels,
            'sequence_lengths': torch.tensor(sequence_lengths, dtype=torch.int),
            'label_lengths': torch.tensor(label_lengths, dtype=torch.int)
        }

        return batch
