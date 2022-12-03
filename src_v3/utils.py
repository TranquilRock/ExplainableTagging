"""Helper functions."""
import random
from typing import Iterable, List

import torch


def set_seed(seed: int) -> None:
    """
    Fixed seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str], to_len: int = None) -> List[int]:
        seq = [self.token_to_id(token) for token in tokens]
        if to_len is not None:
            seq = self.pad_to_len(seq, to_len)
        return seq

    def pad_to_len(self, seq: List[int], to_len: int) -> List[List[int]]:
        pad_seq = seq + [self.pad_id] * max(0, to_len - len(seq))
        return pad_seq
