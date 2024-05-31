from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
import lightning as L
from utils import describe_model_size
from load_dataset import CHAR_PAD, load_vocab


class CNNLayer(nn.Module):
    """CNN + Batch Normalization"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(CNNLayer, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU()

    def _calc_new_sequence_length(self, sequence_lengths: torch.IntTensor):
        """ Calculates output senquence lengths based on CNN receptive field. """
        p = self.cnn.padding[1]
        f = self.cnn.kernel_size[1]
        s = self.cnn.stride[1]
        sequence_lengths = (sequence_lengths + (2 * p) - f) // s + 1
        return sequence_lengths

    def forward(self, features, sequence_lengths: torch.IntTensor):
        sequence_lengths = self._calc_new_sequence_length(sequence_lengths)
        x = features # (batch_size, in_channels, features, time)
        x = self.cnn(x) # (batch_size, out_channels, features, time)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x, sequence_lengths


class RNNLayer(nn.Module):
    """RNN + Layer Normalization"""

    def __init__(self, in_features, hidden_units):
        super(RNNLayer, self).__init__()
        self.rnn = nn.GRU(in_features, hidden_units, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_units)

    def forward(self, features, sequence_lengths: torch.IntTensor):
        x = features # (batch_size, time, in_features)

        # Packed RNN
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, sequence_lengths.cpu(), batch_first=True)
        rnn_output, _hidden_states = self.rnn(rnn_input) 
        x, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, total_length=x.shape[1], batch_first=True)

        x = self.layer_norm(x) # (batch_size, time, 2 * hidden_units)
        return x


@dataclass
class DeepSpeech2Config:
    """Deep Speech 2 Config"""

    # Model-specifig parameters
    vocab: dict[str, int]
    blank_token: str
    n_mels: int # Number of mel filterbanks
    hidden_units: int # Number of RNN hidden units

    # Hyper-parameters
    learning_rate: float = 1e-4

    def n_vocab(self) -> int:
        return len(self.vocab)

    def blank_index(self) -> int:
        return self.vocab[self.blank_token]


class DeepSpeech2(nn.Module):
    """Deep Speech 2.0 Speech Recognition Model"""

    def  __init__(self, config: DeepSpeech2Config):
        super(DeepSpeech2, self).__init__()
        self.config = config

        # CNN Layers
        self.cnn_layers = nn.ModuleList([
            CNNLayer(1, 32, kernel_size=(11, 11), stride=(2, 2)),
            CNNLayer(32, 32, kernel_size=(11, 11), stride=(1, 1), padding=(5, 0)),
            CNNLayer(32, 96, kernel_size=(11, 11), stride=(1, 1), padding=(5, 0)),
        ])
        self.n_features = (self.config.n_mels - 11) // 2 + 1 # Number of features after CNN layers

        # RNN Layers
        self.rnn_layers = nn.ModuleList([
            RNNLayer(
                in_features=96 * self.n_features if i == 0 else 2 * self.config.hidden_units,
                hidden_units=self.config.hidden_units
            ) for i in range(6)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.config.hidden_units, self.config.hidden_units),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_units, self.config.n_vocab())
        )

    def forward(
        self,
        features: torch.FloatTensor,
        sequence_lengths: torch.IntTensor # Sorted in decreasing order
    ) -> torch.FloatTensor:
        x = features # (batch_size, 1, features, time)

        # Apply CNN layers
        for cnn_layer in self.cnn_layers:
            x, sequence_lengths = cnn_layer(x, sequence_lengths)

        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]) # (batch_size, 96 * n_features, time)
        x = x.transpose(1, 2) # (batch_size, time, 96 * n_features)

        # Apply RNN layers
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x, sequence_lengths) # (batch_size, time, 2 * hidden_units)

        logits = self.classifier(x) # (batch_suze, time, n_vocab)
        return logits, sequence_lengths


@dataclass
class ASROutput:
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    metrics: Optional[dict[str, float]] = None


class LightDeepSpeech2(L.LightningModule):
    """A Lightning Wrapper Over ASR Models"""

    def __init__(self, config: DeepSpeech2Config):
        super().__init__()
        self.config = config
        self.asr_model = DeepSpeech2(config)

    def forward(self, batch: dict[str, torch.Tensor], with_metrics: bool = False) -> ASROutput:
        # Forward pass
        batch = {k:v.to(self.device) for k,v in batch.items()}
        logits, sequence_lengths = self.asr_model(batch['features'], batch['sequence_lengths']) # (batch_size, time, n_vocab), (batch_size)

        # Compute loss
        loss = None
        if 'labels' in batch:
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1) # (time, batch_size, n_vocab)
            loss_fn = nn.CTCLoss(self.config.blank_index())
            loss = loss_fn(log_probs, batch['labels'], sequence_lengths, batch['label_lengths'])

        # Compute metrics 
        metrics = self.compute_metrics(logits) if with_metrics else None

        return ASROutput(logits=logits, loss=loss, metrics=metrics)

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx):
        asr_output = self.forward(batch)
        self.log('train_loss', asr_output.loss)
        return asr_output.loss

    def validation_step(self, batch: dict[str, torch.Tensor], _batch_idx):
        asr_output = self.forward(batch, with_metrics=True)
        self.log("val_loss", asr_output.loss, prog_bar=True)
        for metric_name, metric_value in asr_output.metrics.items():
            self.log(f"val_{metric_name}", metric_value, prog_bar=True)

    def compute_metrics(self, logits: torch.Tensor):
        return {'wer': 0, 'cer': 0}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        }
        return [optimizer], [lr_scheduler]


if __name__ == '__main__':
    vocab = load_vocab()
    config = DeepSpeech2Config(
        vocab=vocab,
        blank_token=CHAR_PAD,
        n_mels=80,
        hidden_units=512,
    )
    describe_model_size(DeepSpeech2(config))
