import numpy as np
from dataclasses import dataclass
from vocab import Vocab


@dataclass
class Tokenizer:
    """CTC Tokenizer"""

    vocab: Vocab

    def tokenize(self, sentences: list[str], add_sos=False, add_eos=False) -> list[list[int]]:
        pre = [self.vocab.sos_idx()] if (add_sos and self.vocab.sos_token is not None) else []
        post = [self.vocab.eos_idx()] if (add_eos and self.vocab.eos_token is not None) else []
        return [pre + [self.vocab.char_to_idx[c] for c in s] + post for s in sentences]

    def decode(
        self,
        indices: np.ndarray, # (batch_size, time)
    ) -> list[str]:
        # Replace ignored label indices with pad token
        indices[indices == -100] = self.vocab.blank_idx()

        preds = [''.join([self.vocab.idx_to_char[idx] for idx in idx_list if idx != self.vocab.blank_idx()]) for idx_list in indices]

        # Handle SOS and EOS tokens if they exist in the vocab
        if self.vocab.eos_token is not None:
            preds = [pred.split(self.vocab.eos_token)[0] for pred in preds]
        if self.vocab.sos_token is not None:
            sos = self.vocab.sos_token
            preds = [(p if not p.startswith(sos) else p.replace(sos, '', 1)) for p in preds]

        return preds
