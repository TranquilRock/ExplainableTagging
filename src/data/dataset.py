from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import transformers
from torch.utils.data import Dataset

CLS = 101
SEP = 102
RELEVANT = 1
IRRELEVANT = 2

class RelationalDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, List[Dict[str, str]]],
        tokenizer,
        mode: Literal["train", "dev", "test"],
        max_length: int,  
    ):
        data = self._preprocess(data)
        data = self._tokenize(data, tokenizer)
        self.data = data
        self.mode = mode
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        
        q_r_seqs = []
        for q_ids in self.data[idx]['q']['input_ids']:
            r_max_length = self.max_length - 5 - len(q_ids)

            total_length = 0
            rs = []
            for r_ids in self.data[idx]['r']['input_ids']:
                if total_length + len(r_ids) > r_max_length:
                    break
                rs += r_ids
                total_length += len(r_ids)

            padding_length = r_max_length - total_length
            q_r_seq = [CLS] + [RELEVANT if self.data[idx]['s'] else IRRELEVANT] + [SEP] + q_ids + [SEP] + rs + [SEP] + [0] * padding_length
            q_r_seqs.append(torch.tensor(q_r_seq))
        
        r_q_seqs = []
        for r_ids in self.data[idx]['r']['input_ids']:
            q_max_length = self.max_length - 5 - len(r_ids)

            total_length = 0
            qs = []
            for q_ids in self.data[idx]['q']['input_ids']:
                if total_length + len(q_ids) > q_max_length:
                    break
                qs += q_ids
                total_length += len(q_ids)

            padding_length = q_max_length - total_length
            r_q_seq = [CLS] + [RELEVANT if self.data[idx]['s'] else IRRELEVANT] + [SEP] + r_ids + [SEP] + qs + [SEP] + [0] * padding_length
            r_q_seqs.append(torch.tensor(r_q_seq))
            
        return q_r_seqs, self.data[idx]['q_ans'], r_q_seqs, self.data[idx]['r_ans'] 

    def _preprocess(self, data: List[Dict[str, Union[str, bool, List[str]]]]) -> List[Dict[str, Any]]:
        """Add answer field."""
        for entry in data:
            q_ans = [False] * len(entry['q'])
            for i, q in enumerate(entry['q']):
                for qq in entry['qq']:
                    if qq in q:
                        q_ans[i] = True
            entry['q_ans'] = q_ans

            r_ans = [False] * len(entry['r'])
            for i, r in enumerate(entry['r']):
                for rr in entry['rr']:
                    if rr in r:
                        r_ans[i] = True
            entry['r_ans'] = r_ans
            
        return data

    def _tokenize(self, data: List[Dict[str, Any]], tokenizer: transformers.PreTrainedTokenizer) -> List[Dict[str, Any]]:
        for i, entry in enumerate(data):
            data[i]['q'] = tokenizer(entry['q'], add_special_tokens=False)
            data[i]['r'] = tokenizer(entry['r'], add_special_tokens=False)
            data[i]['qq'] = tokenizer(entry['qq'], add_special_tokens=False)
            data[i]['rr'] = tokenizer(entry['rr'], add_special_tokens=False)
            
        return data

