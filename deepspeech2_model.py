import torch
from torch import nn
from utils import describe_model_size


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


class DeepSpeech2(nn.Module):
    """Deep Speech 2.0 Speech Recognition Model"""

    def  __init__(
        self,
        n_vocab: int, # Vocabulary size
        n_features: int, # Number of input features, i.e., n_mels
        hidden_units: int, # Number of RNN hidden units
    ):
        super(DeepSpeech2, self).__init__()
        self.hidden_units = hidden_units
        self.n_vocab = n_vocab

        # CNN Layers
        self.cnn_layers = nn.ModuleList([
            CNNLayer(1, 32, kernel_size=(11, 41), stride=(2, 2)),
            CNNLayer(32, 32, kernel_size=(11, 21), stride=(1, 2), padding=(5, 0)),
            CNNLayer(32, 96, kernel_size=(11, 21), stride=(1, 2), padding=(5, 0)),
        ])
        self.n_feaures = (n_features - 11) // 2 + 1 # Number of features after CNN layers

        # RNN Layers
        self.rnn_layers = nn.ModuleList([
            RNNLayer(
                in_features=96 * self.n_feaures if i == 0 else 2 * self.hidden_units,
                hidden_units=self.hidden_units
            ) for i in range(6)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_units, self.hidden_units),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.n_vocab)
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
        return logits


if __name__ == '__main__':
    describe_model_size(DeepSpeech2(40, 81, 512))
