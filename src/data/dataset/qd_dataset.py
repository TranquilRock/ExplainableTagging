"""TODO"""
import random
from typing import Any, Dict, List, Literal, Tuple

import torch
from torch.utils.data import Dataset


class QDDataset(Dataset):
    """TODO"""

    def __init__(
        self,
        data: List[Dict],
        vocab,
        query_max_length: int,
        document_max_length: int,
        mode: str,
    ):
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
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

    def _preprocess(self, data: List[Dict[str, Any]]) -> List[Any]:
        data_list = []
        for element in data:
            pid = element["id"]
            art_q = [w.lower() for w in element["q"]]
            art_r = [w.lower() for w in element["r"]]
            agree = [element["s"].lower()]
            rel_q = element["q_labels"] if self.mode == "train" else []
            rel_r = element["r_labels"] if self.mode == "train" else []
            data_list += [
                self._formulate_item(
                    art_q, art_r, agree, rel_q, pid, element["q"], "q"
                ),
                self._formulate_item(
                    art_r, art_q, agree, rel_r, pid, element["r"], "r"
                ),
            ]
        return data_list

    def _formulate_item(
        self,
        art: List[str],
        qry: List[str],
        agree: List[str],
        rel_art: List[int],
        pid: str,
        raw_art: List[str],
        split: Literal["q", "r"],
    ) -> Tuple:
        article = art[0 : 0 + self.query_max_length]
        article = self.vocab.encode(article, self.query_max_length)

        mid = (0 + len(qry)) // 2 + random.randint(-64, 64)
        query_start = max(
            0,
            min(
                mid - self.document_max_length // 2, len(qry) - self.document_max_length
            ),
        )
        query = qry[query_start : query_start + self.document_max_length]
        query = self.vocab.encode(query, self.document_max_length)

        agree = self.vocab.encode(agree)
        query_tokens = torch.LongTensor(agree + query)
        article_tokens = torch.LongTensor(article)

        if self.mode == "train":
            relevant_article = rel_art[: self.query_max_length]
            relevant_article = self._pad_to_len(relevant_article, self.query_max_length)
            relevant_article = torch.LongTensor(relevant_article)
            return (query_tokens, article_tokens, relevant_article)
        else:
            return (
                pid,
                split,
                query_tokens,
                article_tokens,
                raw_art[: self.query_max_length],
            )
