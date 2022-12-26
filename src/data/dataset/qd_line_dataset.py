"""TODO"""
import random
from typing import Any, Dict, List, Literal, Set, Tuple, Union

import torch
from torch.utils.data import Dataset


class QDLineDataset(Dataset):
    """Use data_v2"""

    def __init__(
        self,
        data: Dict[str, Any],
        vocab,
        query_max_length: int,
        document_max_length: int,
        mode: str,
    ):
        self.art_max_len = document_max_length
        self.qry_max_len = query_max_length
        self.mode = mode
        self.vocab = vocab
        self.data_list = self._preprocess(data)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Any:
        return self.data_list[idx]

    def _pad_to_len(self, seq: List[int], to_len: int) -> List[int]:
        pad_seq = seq + [0] * max(0, to_len - len(seq))
        return pad_seq

    def _get_rel(self, art: List[str], rel: List[str]) -> Set[int]:
        is_rel = [False] * len(art)
        for i, line in enumerate(art):
            for rel_line in rel:
                if rel_line in line:
                    is_rel[i] = True
                    break
        return set([i for i, r in enumerate(is_rel) if r])

    def _preprocess(self, data: Dict[str, Dict[str, Union[List[str], bool]]]) -> List[Any]:
        data_list = []
        for pid, ele in data.items():
            art_q = [l.lower() for l in ele["q"]]  # List of lines
            art_r = [l.lower() for l in ele["r"]]  # List of lines
            agree = ['agree' if ele["s"] else 'disagree']

            rel_q = set()
            rel_r = set()
            if self.mode == "train":
                rel_q = self._get_rel(art_q, [l.lower() for l in ele["qq"]])
                rel_r = self._get_rel(art_r, [l.lower() for l in ele["rr"]])

            data_list += [
                self._formulate_item(
                    art_q, art_r, agree, rel_q, pid, ele["q"], "q"
                ),
                self._formulate_item(
                    art_r, art_q, agree, rel_r, pid, ele["r"], "r"
                ),
            ]
        return data_list

    def _formulate_item(
        self,
        art: List[str],
        qry: List[str],
        agree: List[str],
        rel_idx: Set[int],
        pid: str,
        raw_art: List[str],
        split: Literal["q", "r"],
    ) -> Tuple:

        article = []
        article_idx = []
        label = []
        for i, line in enumerate(art):
            sep_line = line.split(' ')
            article_idx += [i] * len(sep_line)
            article += sep_line
            label += ([1] if i in rel_idx else [0]) * len(sep_line)
        # ========== Resplit twice, try to accelerate ===========
        query = []
        for line in qry:
            query += line.split(' ')

        article = self.vocab.encode(
            article[:self.art_max_len],
            self.art_max_len,
        )
        article_idx = article_idx[:self.art_max_len]
        query = self.vocab.encode(
            query[:self.qry_max_len],
            self.qry_max_len,
        )
        agree = self.vocab.encode(agree)
        query_tokens = torch.LongTensor(agree + query)
        article_tokens = torch.LongTensor(article)

        if self.mode == "train":
            label = self._pad_to_len(
                label[: self.art_max_len], self.art_max_len)
            label = torch.LongTensor(label)

            return (query_tokens, article_tokens, label)
        else:
            return (
                pid,
                split,
                query_tokens,
                article_tokens,
                raw_art,
                article_idx,
            )
