import csv
import random

import torch
from typing import Dict, List, Any

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


def _read_data(data_path: str) -> Dict[str, Dict[str, Any]]:
    """Reads from data_path
    data: Dict[str, Dict[str, Any]]
        = {
            '0': {
                    'q': str,
                    'r': str,
                    's': str,
                    'qq': List[str],
                    'rr': List[str],
                 },
            ...
           }
    """
    data = {}
    with open(data_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            id, q, r, s, qq, rr = (
                row["id"],
                row["q"],
                row["r"],
                row["s"],
                row["q'"],
                row["r'"],
            )
            if id not in data:
                data[id] = {
                    "q": q,
                    "r": r,
                    "s": s,
                    "qq": [],
                    "rr": [],
                }
            data[id]["qq"].append(qq)
            data[id]["rr"].append(rr)
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
) -> Dict[str, Dict[str, Any]]:
    """
    Read, preprocess and tokenize data
    data: Dict[str, Dict[str, Any]]
        = {
            '0': {
                    'q': str,
                    'r': str,
                    's': str,
                    'qq': List[Dict[str, List]],
                    'rr': List[Dict[str, List]],
                 },
            ...
           }
    """
    # Read data
    data = _read_data(data_path)

    # Preprocess
    data = _preprocess(data)

    # tokenize: q, r, q', r'
    for _, entry in data.items():
        entry["q"] = tokenizer(entry["q"], add_special_tokens=False)
        entry["r"] = tokenizer(entry["r"], add_special_tokens=False)
        entry["qq"] = [tokenizer(qq, add_special_tokens=False) for qq in entry["qq"]]
        entry["rr"] = [tokenizer(qq, add_special_tokens=False) for qq in entry["rr"]]
    return data
