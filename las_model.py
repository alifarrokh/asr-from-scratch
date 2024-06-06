from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch import nn
from evaluate import load as load_metric
import lightning as L
from vocab import Vocab
from tokenizer import Tokenizer
from utils import describe_model_size


@dataclass
class LASConfig:
    """LAS model config"""

    n_mels: int # Number of mel filterbanks
    hidden_units: int # Number of RNN hidden units
    tokenizer: Tokenizer
    max_output_length: int

    # TODO -> Improve the config classes
    learning_rate: float = 1e-4

    def __post_init__(self):
        assert self.tokenizer.vocab.sos_token is not None, "SOS token is required in LAS" 
        assert self.tokenizer.vocab.eos_token is not None, "EOS token is required in LAS" 


class PBLSTM(nn.Module):
    """
    Pyramidal Bidirectional LSTM

    Note: If reduction=True, input sequence length must be divisible by two.
    """

    def __init__(
        self,
        input_feature_dim: int,
        hidden_units: int,
        reduction: bool = True,
    ):
        super(PBLSTM, self).__init__()
        self.reduction = reduction

        self.lstm = nn.LSTM(
            input_size=2 * input_feature_dim if reduction else input_feature_dim,
            hidden_size=hidden_units,
            bidirectional=True,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.FloatTensor,
        sequence_lengths: torch.IntTensor # Sorted in decreasing order
    ) -> tuple[torch.FloatTensor, torch.IntTensor]:
        batch_size, time_dim, input_feature_dim = x.size()

        # Dimension reduction
        total_length = time_dim
        if self.reduction:
            x = x.contiguous().view(batch_size, time_dim // 2, input_feature_dim * 2)
            sequence_lengths = torch.ceil(sequence_lengths / 2).type(torch.IntTensor)
            total_length //= 2

        # LSTM
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, sequence_lengths.cpu(), batch_first=True)
        rnn_output, _hidden_states = self.lstm(rnn_input) 
        x, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, total_length=total_length, batch_first=True)
        return x, sequence_lengths


class Listener(nn.Module):
    """LAS's Listener Module"""

    def __init__(self, config: LASConfig):
        super(Listener, self).__init__()
        self.config = config

        self.lstm0 = PBLSTM(config.n_mels, config.hidden_units, reduction=False)
        self.lstm1 = PBLSTM(config.hidden_units * 2, config.hidden_units)
        self.lstm2 = PBLSTM(config.hidden_units * 2, config.hidden_units)
        self.lstm3 = PBLSTM(config.hidden_units * 2, config.hidden_units)

    def forward(
        self,
        x: torch.FloatTensor, # (batch_size, time, n_mels)
        sequence_lengths: torch.IntTensor # Sorted in decreasing order
    ) -> tuple[torch.FloatTensor, torch.IntTensor]:
        x, sequence_lengths = self.lstm0(x, sequence_lengths) # (batch_size, time, hidden_units * 2)
        x, sequence_lengths = self.lstm1(x, sequence_lengths) # (batch_size, time // 2, hidden_units * 2)
        x, sequence_lengths = self.lstm2(x, sequence_lengths) # (batch_size, time // 4, hidden_units * 2)
        x, sequence_lengths = self.lstm3(x, sequence_lengths) # (batch_size, time // 8, hidden_units * 2)
        return x, sequence_lengths


class Speller(nn.Module):
    """LAS's Attention and Speller Module"""

    def __init__(self, config: LASConfig):
        super(Speller, self).__init__()
        self.config = config
        self.vocab_size = len(config.tokenizer.vocab)

        self.lstm = nn.LSTM(self.vocab_size + (config.hidden_units * 2), config.hidden_units * 2, batch_first=True)
        self.phi = nn.Linear(config.hidden_units * 2, config.hidden_units) # Transforms hidden states of the first LSTM layer
        self.psi = nn.Linear(config.hidden_units * 2, config.hidden_units) # Transforms outputs of the listener
        self.character_distribution = nn.Linear(config.hidden_units * 4, self.vocab_size)

    def _attention(
        self,
        x: torch.FloatTensor, # Listener's output features
        state: torch.FloatTensor, # Last LSTM state
        sequence_lengths: torch.IntTensor,
    ) -> torch.FloatTensor:
        time_dim = x.size(1)

        features_repr = self.psi(x) # (batch_size, time_dim, hidden_units)
        state_repr = self.phi(state) # (batch_size, 1, hidden_units)
        att_scores = torch.bmm(features_repr, state_repr.transpose(1, 2)) # (batch_size, time_dim, 1)

        att_mask = torch.tensor([[1] * l + [0] * (time_dim - l) for l in sequence_lengths], dtype=torch.bool) # (batch_size, time_dim)
        att_mask = att_mask.unsqueeze(-1).to(x.device) # (batch_size, time_dim)
        att_scores = att_scores.masked_fill(~ att_mask, - torch.inf)

        att_weights = nn.functional.softmax(att_scores.squeeze(-1), dim=-1).unsqueeze(1) # (batch_size, 1, time_dim)
        context = torch.bmm(att_weights, x) # (batch_size, 1, hidden_units * 2)
        return context

    def forward(
        self,
        x: torch.FloatTensor,
        sequence_lengths: torch.IntTensor, # Sorted in decreasing order
        labels: torch.IntTensor = None, # (batch_size, max_label_length)
    ) -> torch.FloatTensor:
        batch_size, time_dim, _ = x.size()

        # Convert labels to one hot vectors
        if labels is not None:
            labels_one_hot = labels.clone()
            labels_one_hot[labels_one_hot == -100] = 0
            labels_one_hot = nn.functional.one_hot(labels_one_hot, self.vocab_size)

        last_pred = torch.tensor([self.config.tokenizer.vocab.sos_idx()] * batch_size) # (batch_size)
        last_pred = nn.functional.one_hot(last_pred, self.vocab_size).to(x.device) # (batch_size, vocab_size)
        last_pred = last_pred.float().unsqueeze(1) # (batch_size, 1, vocab_size)

        context = x[:, :1, :] # (batch_size, 1, hidden_units * 2)
        hidden_state = None

        raw_preds = []
        eos_reached = torch.zeros((batch_size, 1), dtype=torch.bool).to(x.device).detach()
        max_length = labels.size(1) if labels is not None else self.config.max_output_length

        for step in range(max_length):
            # LSTM forward pass
            lstm_input = torch.cat([last_pred, context], dim=-1) # (batch_size, 1, n_vocab + hidden_units * 2)
            state, hidden_state = self.lstm(lstm_input, hidden_state)

            # Attention
            context = self._attention(x, state, sequence_lengths)

            # Prediction
            char_dist_input = torch.cat([context.squeeze(1), state.squeeze(1)], dim=-1) # (batch_size, hidden_units * 4)
            raw_pred = self.character_distribution(char_dist_input) # (batch_size, vocab_size)

            # Sampling
            if labels == None or np.random.rand() <= 0.1:
                last_pred = nn.functional.softmax(raw_pred, dim=-1).unsqueeze(1) # (batch_size, 1, vocab_size)

                # TODO ->  Fix this! used for debugging
                xx = torch.zeros_like(last_pred)
                for i in range(xx.shape[0]):
                    xx[i, 0, last_pred[i, 0].argmax()] = 1
                last_pred = xx

            # Teacher Forcing
            else:
                last_pred = labels_one_hot[:, step:step+1, :]

            raw_preds.append(raw_pred.unsqueeze(1))

            # Check if all sequences are finished (reached EOS token)
            if not self.training and labels == None:
                is_eos = raw_pred.argmax(dim=-1, keepdim=True) == self.config.tokenizer.vocab.eos_idx()
                eos_reached = torch.logical_or(eos_reached, is_eos).detach()
                if eos_reached.sum() == batch_size:
                    break

        raw_preds_tensor = torch.cat(raw_preds, dim=1) # (batch_size, <= max_length, vocab_size)
        return raw_preds_tensor


class LAS(nn.Module):
    """LAS Speech Recognition Model"""

    def __init__(self, config: LASConfig):
        super(LAS, self).__init__()
        self.config = config

        self.listener = Listener(config)
        self.speller = Speller(config)

    def forward(
        self,
        x: torch.FloatTensor,
        sequence_lengths: torch.IntTensor, # Sorted in decreasing order
        labels: torch.IntTensor = None, # (batch_size, max_label_length)
    ) -> torch.FloatTensor:
        x = x.transpose(1, 2) # (batch_size, time, n_mels)
        audio_repr, sequence_lengths = self.listener(x, sequence_lengths)
        raw_preds = self.speller(audio_repr, sequence_lengths, labels)
        return raw_preds


@dataclass
class LASOutput:
    raw_preds: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    metrics: Optional[torch.Tensor] = None


class LightLAS(L.LightningModule):
    """A Lightning Wrapper Over LAS"""

    def __init__(self, config: LASConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.asr_model = LAS(config)
        self.wer_metric = load_metric("wer")
        self.cer_metric = load_metric("cer")

    def forward(self, batch: dict[str, torch.Tensor], with_metrics: bool = False) -> LASOutput:
        # Forward pass
        batch = {k:v.to(self.device) for k,v in batch.items()}
        raw_preds = self.asr_model(
            batch['features'],
            batch['sequence_lengths'],
            batch.get('labels', None),
        ) # (batch_size, time, vocab_size)

        loss = None
        metrics = None
        if 'labels' in batch:
            loss = nn.CrossEntropyLoss()(raw_preds.transpose(1, 2), batch['labels'])

            # Compute metrics
            if with_metrics:
                metrics = self.compute_metrics(raw_preds.detach(), batch['labels'])

        return LASOutput(raw_preds=raw_preds, loss=loss, metrics=metrics)

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx):
        asr_output = self.forward(batch)
        self.log('train_loss', asr_output.loss)
        return asr_output.loss

    def validation_step(self, batch: dict[str, torch.Tensor], _batch_idx):
        asr_output = self.forward(batch, with_metrics=True)
        self.log("val_loss", asr_output.loss, prog_bar=True)
        if asr_output.metrics is not None:
            for metric_name, metric_value in asr_output.metrics.items():
                self.log(f"val_{metric_name}", metric_value, prog_bar=True)

    def compute_metrics(self, raw_preds: torch.Tensor, labels: torch.Tensor):
        # Decode predictions
        pred_indices = raw_preds.detach().cpu().numpy().argmax(-1)
        preds = self.config.tokenizer.decode_pred_indices(pred_indices)

        # Decode labels
        labels = labels.cpu().numpy()
        labels[labels == -100] = self.config.tokenizer.vocab.eos_idx()
        labels = self.config.tokenizer.decode_pred_indices(labels)

        wer = self.wer_metric.compute(predictions=preds, references=labels)
        cer = self.cer_metric.compute(predictions=preds, references=labels)
        return {'wer': wer, 'cer': cer}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        }
        return [optimizer], [lr_scheduler]


if __name__ == '__main__':
    vocab = Vocab.from_json('las_vocab.json')
    tokenizer = Tokenizer(vocab)
    config = LASConfig(
        n_mels=60,
        hidden_units=256,
        tokenizer=tokenizer,
        max_output_length=300,
    )
    describe_model_size(LAS(config))
