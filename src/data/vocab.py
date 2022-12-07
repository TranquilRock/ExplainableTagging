"""Vocabulary Dict for token id mapping."""
from typing import Iterable, List, Optional


class Vocab:
    """Vocabulary Dict for token id mapping."""

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
        """Get id for [PAD] token."""
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        """Get id for [UNK] token."""
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        """Get all tokens."""
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        """Transform string token to id, return unk if not exist."""
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str], to_len: Optional[int] = None) -> List[int]:
        """Encode list of tokens and pad to specified length."""
        seq = [self.token_to_id(token) for token in tokens]
        if to_len is not None:
            seq = self.pad_to_len(seq, to_len)
        return seq

    def pad_to_len(self, seq: List[int], to_len: int) -> List[int]:
        """Pad sequence to specified length."""
        pad_seq = seq + [self.pad_id] * max(0, to_len - len(seq))
        return pad_seq
