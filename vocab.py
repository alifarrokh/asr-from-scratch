from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from load_dataset import get_alphabet


DEFALUT_BLK = '[PAD]' # CTC BLank & Padding
DEFAULT_SOS = '[BOS]' # Used in LAS
DEFAULT_EOS = '[EOS]' # Used in LAS
DEFAULT_UNK = '[UNK]' # Used in XLSR


@dataclass
class Vocab:
    """Character-based Speech Recognition Vocabulary for CTC Models"""

    idx_to_char: list[str]
    char_to_idx: dict[str, int] = field(init=False)

    blank_token: str
    sos_token: Optional[str] = None
    eos_token: Optional[str] = None
    unk_token: Optional[str] = None

    def __post_init__(self):
        self.char_to_idx = {c:idx for idx,c in enumerate(self.idx_to_char)}

    def __len__(self) -> int:
        return len(self.idx_to_char)

    def from_alphabet(alphabet: list, blank_token=DEFALUT_BLK, sos_token=None, eos_token=None, unk_token=None):
        assert blank_token is not None, "Blank token is mandatory in CTC"
        assert len(alphabet) > 0, "Alphabet is empty"
        special_chars = [blank_token, sos_token, eos_token, unk_token]
        special_chars = [c for c in special_chars if c is not None]
        return Vocab(
            idx_to_char=special_chars + alphabet,
            blank_token=blank_token,
            sos_token=sos_token,
            eos_token=eos_token,
            unk_token=unk_token
        )

    def save(self, json_path):
        vocab_dict = asdict(self)
        vocab_dict.pop('char_to_idx')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, indent=4)

    def from_json(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        return Vocab(**vocab_dict)

    def blank_idx(self) -> int:
        return self.char_to_idx[self.blank_token]

    def sos_idx(self) -> int:
        return self.char_to_idx[self.sos_token]

    def eos_idx(self) -> int:
        return self.char_to_idx[self.eos_token]


if __name__ == '__main__':
    alphabet = get_alphabet()

    # Generate LAS vocab file
    vocab = Vocab.from_alphabet(alphabet, sos_token=DEFAULT_SOS, eos_token=DEFAULT_EOS)
    vocab.save('las_vocab.json')

    # Generate XLSR vocab file
    vocab = Vocab.from_alphabet(alphabet, unk_token=DEFAULT_UNK)
    vocab.save('xlsr_vocab.json')

    # Generate Deep Speech 2 vocab file
    vocab = Vocab.from_alphabet(alphabet)
    vocab.save('deepspeech2_vocab.json')
