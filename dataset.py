import random
from typing import Dict, List, Literal, Tuple

import torch
from torch.utils.data import Dataset

CLS = 101
SEP = 102


class RelationalDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, List[Dict[str, str]]],
        ids: List[str],
        mode: Literal["train", "valid"],
    ):
        self.data = data
        self.ids = ids
        self.mode = mode

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _id = self.ids[idx]
        _q = self.data[_id]["q"]
        _r = self.data[_id]["r"]
        _qq = random.choice(self.data[_id]["qq"])  # TODO maybe fix this?
        _rr = random.choice(self.data[_id]["rr"])

        return (
            _id,
            torch.tensor(_q["input_ids"]),
            torch.tensor(_r["input_ids"]),
            torch.tensor(_qq["input_ids"]),
            torch.tensor(_rr["input_ids"]),
        )

    def get_max_data(self) -> Tuple[int, str]:
        max_len = float("-inf")
        key = None
        split = None
        for k, v in self.data.items():
            for _split in ("q", "r"):
                l = len(v[_split]["input_ids"])
                if l > max_len:
                    max_len = l
                    key = k
                    split = _split
        return max_len, self.data[key][split]
