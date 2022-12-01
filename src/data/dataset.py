"""Construct dataset from process_raw.jsonv2()."""
from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import transformers
from torch.utils.data import Dataset

from transformers import LongformerConfig


# For longformer
config = LongformerConfig()
SEP = config.sep_token_id
PAD = config.pad_token_id


class LongformerDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, List[Dict[str, str]]],
        tokenizer: transformers.PreTrainedTokenizer,
        mode: Literal["train", "dev", "test"],
        sentence_max_length: int,
        document_max_length: int,
    ):
        self.mode = mode
        self.sentence_max_length = sentence_max_length
        self.document_max_length = document_max_length
        # TODO check the constant of additional TAGs
        self.max_length = self.sentence_max_length + self.document_max_length + 8

        if self.mode == "train":
            data = self._preprocess(data)
        data = self._tokenize(data, tokenizer)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        q_and_r_p_list = []
        for q in self.data[idx]['q']:
            if self.data[idx]['s']:  # already contain SEP in tokens
                q_and_r_p = q + \
                    self.agree_token[1:] + self.data[idx]['r_p'][1:]
            else:
                q_and_r_p = q + \
                    self.disagree_token[1:] + self.data[idx]['r_p'][1:]
            padding = [PAD] * (self.max_length - len(q_and_r_p))
            q_and_r_p = q_and_r_p + padding
            q_and_r_p_list.append(torch.tensor(q_and_r_p))

        r_and_q_p_list = []
        for r in self.data[idx]['r']:
            r_and_q_p = r \
                + (self.agree_token[1:] if self.data[idx]['s']
                   else self.disagree_token[1:]) \
                + self.data[idx]['q_p'][1:]  # Skip the first CLS tag
            padding = [PAD] * (self.max_length - len(r_and_q_p))
            r_and_q_p = r_and_q_p + padding
            r_and_q_p_list.append(torch.tensor(r_and_q_p))

        if self.mode == 'train':
            return q_and_r_p_list, self.data[idx]['q_ans'], r_and_q_p_list, self.data[idx]['r_ans']
        else:
            return q_and_r_p_list, r_and_q_p_list

    def _preprocess(self, data: List[Dict[str, Union[str, bool, List[str]]]]) -> List[Dict[str, Any]]:
        """Append answer field."""
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

    def _tokenize(self,
                  data: List[Dict[str, Any]],
                  tokenizer: transformers.PreTrainedTokenizer) -> List[Dict[str, Any]]:
        self.agree_token = tokenizer('Agree').input_ids
        self.disagree_token = tokenizer('Disagree').input_ids
        for i, entry in enumerate(data):
            data[i]['q_p'] = tokenizer(' '.join(
                entry['q']), max_length=self.document_max_length, truncation=True).input_ids
            data[i]['r_p'] = tokenizer(' '.join(
                entry['r']), max_length=self.document_max_length, truncation=True).input_ids
            data[i]['q'] = tokenizer.batch_encode_plus(
                entry['q'], max_length=self.sentence_max_length, truncation=True).input_ids
            data[i]['r'] = tokenizer.batch_encode_plus(
                entry['r'], max_length=self.sentence_max_length, truncation=True).input_ids

        return data
