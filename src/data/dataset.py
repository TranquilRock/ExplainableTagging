from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import transformers
from torch.utils.data import Dataset

CLS = 101
SEP = 102


class RelationalDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, List[Dict[str, str]]],
        tokenizer,
        mode: Literal["train", "dev", "test"],
    ):
        data = self._preprocess(data)
        data = self._tokenize(data, tokenizer)
        self.data = data
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

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
        for entry in data:
            entry['q'] = tokenizer(entry['q'], add_special_tokens=False)
            entry['r'] = tokenizer(entry['r'], add_special_tokens=False)
            entry['qq'] = tokenizer(entry['qq'], add_special_tokens=False)
            entry['rr'] = tokenizer(entry['rr'], add_special_tokens=False)
        return data
