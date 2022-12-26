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
        mode: str,
    ):
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        self.mode = mode
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
        
    def _formulate_item(self, q, r, s, q_labels, r_labels, pid, rawq, rawr):
        query = q[:self.query_max_length]
        query = self.vocab.encode(query, self.query_max_length)

        document = r[:self.document_max_length]
        document = self.vocab.encode(document, self.document_max_length)

        relation = self.vocab.encode(s)

        input_tokens = torch.LongTensor(query + document + relation)

        if self.mode == 'train':
            query_labels = q_labels[:self.query_max_length]
            query_labels = self._pad_to_len(query_labels, self.query_max_length)
            document_labels = r_labels[:self.document_max_length]
            document_labels = self._pad_to_len(document_labels, self.document_max_length)
            
            labels = torch.LongTensor(query_labels + document_labels)
            return (input_tokens, labels)
        else:
            return (pid, input_tokens, rawq[:self.query_max_length], rawr[:self.query_max_length])
                    
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

            item = self._formulate_item(q, r, s, q_labels, r_labels, pid, element['q'], element['r'])
            data_list.append(item)
            
        return data_list

    
