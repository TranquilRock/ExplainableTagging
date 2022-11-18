import csv
import random

import torch
from typing import Dict, List

from transformers import PreTrainedTokenizer


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


def _read_data(data_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Read data from data path
    data = dict(list(dict()))
         = {'8': [
                     {'id': '8', 'q': '"It can go both ways..."', ... },
                     {'id': '8', 'q': ... },
                     ...
                 ]
            '9': [
                     {'id': '9', 'q': '"I personly would not condone an abortion..."', ... },
                     {'id': '9', 'q': ... },
                     ...
                 ]
            ...
           }
    """
    data = {}
    with open(data_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            if row["id"] not in data:
                data[row["id"]] = [row]
            else:
                data[row["id"]].append(row)

    return data


def _preprocess(
    data: Dict[str, List[Dict[str, str]]]
) -> Dict[str, List[Dict[str, str]]]:
    """
    TODO Preprocess data
    """

    return data


def get_data(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Read, preprocess and tokenize data
    data = dict(list(dict()))
         = {'8': [
                     {'id': '8', 'q': tokenize("It can go both ways..."), ... },
                     {'id': '8', 'q': ... },
                     ...
                 ]
            '9': [
                     {'id': '9', 'q': tokenize("I personly would not condone an abortion..."), ... },
                     {'id': '9', 'q': ... },
                     ...
                 ]
            ...
           }
    """
    # Read data
    data = _read_data(data_path)

    # Preprocess
    data = _preprocess(data)

    # tokenize: q, r, q', r'
    for _, v in data.items():
        for cand in v:
            cand["q"] = tokenizer(cand["q"], add_special_tokens=False)
            cand["r"] = tokenizer(cand["r"], add_special_tokens=False)
            cand["q'"] = tokenizer(cand["q'"], add_special_tokens=False)
            cand["r'"] = tokenizer(cand["r'"], add_special_tokens=False)

    return data
