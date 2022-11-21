import random
from typing import Dict, List, Literal, Tuple

import torch
from torch.utils.data import Dataset

from collections import defaultdict

CLS = 101
SEP = 102


class RelationalDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, List[Dict[str, str]]],
        ids: List[str],
        tokenizer, 
        mode: Literal["train", "valid"],
    ):
        data = self._preprocess(data)
        data = self._tokenize(data, tokenizer)
        data = self._reconstruct(data)
        self.data = data
        self.ids = ids
        self.mode = mode

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index):
        ID = self.ids[index]
        q = self.data[ID][0]['q']
        r = self.data[ID][0]['r']
        return ID, q, r
    
    def _preprocess(self, data):
        """
        Preprocess data
        data: List[dict{str, Union(str, bool, List[str])}]
        = {
            {
               'id': int,
               'q': List[str],
               'r': List[str],
               's': bool,
               'qq': List[str],
               'rr': List[str],
               'qb': List[bool],
               'rb': List[bool],
            },
            ...
          }
        """
        for entry in data:
            entry['qb'] = [0 for i in range(len(entry['q']))]
            for q, qb in zip(entry['q'], entry['qb']):
                for qq in entry['qq']:
                    if qq in q:
                        qb = 1
            entry['rb'] = [0 for i in range(len(entry['r']))]
            for r, rb in zip(entry['r'], entry['rb']):
                for rr in entry['rr']:
                    if rr in r:
                        rb = 1

        return data
    
    def _tokenize(self, data, tokenizer):
        for entry in data:
            entry['q'] = tokenizer(entry['q'], add_special_tokens=False)
            print(entry['id'])
            entry['r'] = tokenizer(entry['r'], add_special_tokens=False)
            entry['qq'] = tokenizer(entry['qq'], add_special_tokens=False)
            entry['rr'] = tokenizer(entry['rr'], add_special_tokens=False)

        return data

    def _reconstruct(self, data):
        new_data = defaultdict(list)
        for d in data:
            new_data[d['id']].append(d)

        return new_data
