"""Helper functions."""
import random
from typing import List

import torch


def set_seed(seed: int) -> None:
    """
    Fixed seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def concat_child(answer_list: List[List[str]]) -> List[List[str]]:
    """
    Originally the structure looks like
        [
            ["Answer line1", "Answer line2", ...]
        ]
    Transform into:
        [
            ["Answer", "line1", "Answer, "line2", ...]
        ]
    For LCS matching.
    """
    for i in range(len(answer_list)):
        tmp_entry = []
        for line in answer_list[i]:
            tmp_entry += line.split(' ')
        answer_list[i] = tmp_entry
    return answer_list
