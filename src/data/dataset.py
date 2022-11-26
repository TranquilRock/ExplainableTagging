from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import transformers
from torch.utils.data import Dataset

class RelationalDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, List[Dict[str, str]]],
        tokenizer,
        mode: Literal["train", "dev", "test"],
        max_length: int,  
    ):
        self.mode = mode
        self.max_length = max_length
        if self.mode == "train":
            data = self._preprocess(data)
        data = self._tokenize(data, tokenizer)
        self.data = data
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.data[idx]['s']:
            s = torch.tensor([0])
        else:
            s = torch.tensor([1])
            
        if self.mode == 'train':
            return self.data[idx]['q'], self.data[idx]['r_p'], self.data[idx]['q_ans'], self.data[idx]['r'], self.data[idx]['q_p'], self.data[idx]['r_ans'], s
        else:
            return self.data[idx]['id'], self.data[idx]['q'], self.data[idx]['r_p'], self.data[idx]['r'], self.data[idx]['q_p'], s
            
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
            data[i]['q_p'] = tokenizer(' '.join(entry['q']), max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt").input_ids
            data[i]['r_p'] = tokenizer(' '.join(entry['r']), max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt").input_ids
            data[i]['q'] = tokenizer.batch_encode_plus(entry['q'], max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt").input_ids
            data[i]['r'] = tokenizer.batch_encode_plus(entry['r'], max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt").input_ids
            
        return data
