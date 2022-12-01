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


class LongformerDatasetV2(Dataset):
    """Input should be data-v2"""
    def __init__(
        self,
        raw: Dict[str, Any],
        tokenizer: transformers.PreTrainedTokenizer,
        mode: Literal["train", "dev", "test"],
        sentence_max_length: int,
        document_max_length: int,
    ):
        self.mode = mode
        self.sentence_max_length = sentence_max_length
        self.document_max_length = document_max_length
        self.agree_token = tokenizer(
            'Agree').input_ids[1:]  # Skips the first CLS tag
        self.disagree_token = tokenizer(
            'Disagree').input_ids[1:]  # Skips the first CLS tag
        # TODO check the constant of additional TAGs
        self.max_length = self.sentence_max_length + self.document_max_length + 8
        # =========================================================
        self.data_list = self._preprocess(raw)
        self.data = self._tokenize(raw, tokenizer)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, bool]:
        pid, split, sentense_id, is_ans = self.data_list[idx]

        ret = self.data[pid][split][sentense_id]
        ret += self.agree_token if self.data[pid]['s'] else self.disagree_token
        ret += self.data[pid]['rp' if split == 'q' else 'qp']
        ret += [PAD] * (self.max_length - len(ret))
        ret = torch.tensor(ret)

        return ret, is_ans

    def _preprocess(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Append answer field."""
        data_list = []
        for k in data.keys():
            if self.mode == "train":
                for i, q in enumerate(data[k]['q']):
                    in_ans = False
                    for qq in data[k]['qq']:  # Check if marked by anyone.
                        if qq in q:
                            in_ans = True
                            break
                    data_list.append((data[k]['id'], 'q', i, in_ans))
                for i, r in enumerate(data[k]['r']):
                    in_ans = False
                    for rr in data[k]['rr']:
                        if rr in r:
                            in_ans = True
                            break
                    data_list.append((data[k]['id'], 'r', i, in_ans))
                del data[k]['qq']
                del data[k]['rr']
            else:
                for i, q in enumerate(data[k]['q']):
                    data_list.append((data[k]['id'], 'q', i, False))
                for i, r in enumerate(data[k]['r']):
                    data_list.append((data[k]['id'], 'r', i, False))
        return data_list

    def _tokenize(self,
                  data: Dict[str, Any],
                  tokenizer: transformers.PreTrainedTokenizer) -> List[Dict[str, Any]]:

        for k in data.keys():
            # Encode q as paragraph and skip first CLS
            data[k]['qp'] = tokenizer(' '.join(
                data[k]['q']), max_length=self.document_max_length, truncation=True).input_ids[1:]
            # Encode r as paragraph and skip first CLS
            data[k]['rp'] = tokenizer(' '.join(
                data[k]['r']), max_length=self.document_max_length, truncation=True).input_ids
            data[k]['q'] = tokenizer.batch_encode_plus(
                data[k]['q'], max_length=self.sentence_max_length, truncation=True).input_ids
            data[k]['r'] = tokenizer.batch_encode_plus(
                data[k]['r'], max_length=self.sentence_max_length, truncation=True).input_ids

        return data
