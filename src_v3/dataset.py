from typing import Any, Dict, List, Literal, Tuple

import torch
from torch.utils.data import Dataset
import random
from utils import Vocab

class SeqtoSeqDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        query_max_length: int,
        document_max_length: int,
        num_classes: int,
        mode: str,
    ):
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        self.mode = mode
        self.num_classes = num_classes
        self.vocab = vocab
        self.data_list = self._preprocess(data)
        
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            input_tokens, query_labels = self.data_list[index]
            return input_tokens, query_labels
        else:
            pid, split, input_tokens, raw_query = self.data_list[index]
            return pid, split, input_tokens, raw_query

    def _pad_to_len(self, seq: List[int], to_len: int) -> List[List[int]]:
        pad_seq = seq + [0] * max(0, to_len - len(seq))
        return pad_seq
        
    def _formulate_item(self, x, y, s, x_labels, pid, rawx, split):
        for i in range(0, len(x), self.query_max_length):
            query = x[i:i+self.query_max_length]
            query = self.vocab.encode(query, self.query_max_length)

            mid = (0 + len(y)) // 2 + random.randint(-64, 64)
            document_start = max(0, min(mid - self.document_max_length // 2, len(y) - self.document_max_length))
            document_end = document_start + self.document_max_length
            document = y[document_start:document_end]
            document = self.vocab.encode(document, self.document_max_length)

            relation = self.vocab.encode(s)

            input_tokens = torch.LongTensor(query + relation + document)

            if self.mode == 'train':
                query_labels = x_labels[i:i+self.query_max_length]
                query_labels = self._pad_to_len(query_labels, self.query_max_length)
                query_labels = [[0, 1] if label == 1 else [1, 0] for label in query_labels ]
                query_labels = torch.FloatTensor(query_labels)
                return (input_tokens, query_labels)
            else:
                return (pid, split, input_tokens, rawx[i:i+self.query_max_length])
                    
    def _preprocess(self, data: List[Dict]):
        data_list = []
        for element in data:
            pid = element['id']
            q = [w.lower() for w in element['q']]
            r = [w.lower() for w in element['r']]
            s = [element['s'].lower()]
            if self.mode == 'train':
                q_labels = element['q_labels']
                r_labels = element['r_labels']
            else:
                q_labels = []
                r_labels = []

            item = self._formulate_item(q, r, s, q_labels, pid, element['q'], 'q')
            data_list.append(item)
            item = self._formulate_item(r, q, s, r_labels, pid, element['r'], 'r')
            data_list.append(item)
            
        return data_list

    
