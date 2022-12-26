"""Data utility to process raw data into proper form.
This file shall not be used directly.
"""
import csv
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from random import random
from typing import Any, Dict, List, Union

import torch
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from data import Vocab

DATA_ROOT = "../data/"


def data_v1(data_path: str) -> List[Dict[str, Union[int, bool, List[str]]]]:
    """Reads raw data file from data_path
    data = [{
            'id': 8,
            'q': [
                'It can go both ways .',
                'We all doubt .',
                'It is what you do with it that matters .'
            ],
            'r': ['True .'],
            's': True,
            'qq': ['It can go both ways', 'We all doubt', 'It is what you do with it that matters'],
            'rr': ['True'],
            }, ]
    """
    data = []
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            pid = int(row["id"])
            s = row["s"] == "AGREE"
            q = row["q"].strip('"').split(" .")
            q = [line.strip(" ") + " ." for line in q if line != ""]
            r = row["r"].strip('"').split(" .")
            r = [line.strip(" ") + " ." for line in r if line != ""]
            qq = row["q'"].strip('"').split(" .")
            qq = [line.strip(" ") for line in qq if line != ""]
            rr = row["r'"].strip('"').split(" .")
            rr = [line.strip(" ") for line in rr if line != ""]
            data.append(
                {
                    "id": pid,
                    "q": q,
                    "r": r,
                    "s": s,
                    "qq": qq,
                    "rr": rr,
                }
            )
    return data


def data_v2(data_path: str, is_test: bool = False) -> Dict[str, Dict[str, Any]]:
    """Result looks like:
    {
        "8": {
            "q": [
                "It can go both ways .",
                "We all doubt .",
                "It is what you do with it that matters ."
            ],
            "r": [
                "True ."
            ],
            "s": true,
            "qq": [
                "It can go both ways",
                "We all doubt",
                "It is what you do with it that matters",
                "can go both ways",
                "We all doubt",
                "It is what you do with it that matters",
                "It can go both ways",
                "We all doubt",
                "It is what you do with it that matters"
            ],
            "rr": [
                "True",
                "True",
                "True"
            ]
        },
        # ...
    }
    """
    data = {}
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in tqdm(reader):
            pid = row["id"]
            s = (row["s"] == "AGREE") if "s" in row else False
            q = row["q"].strip('"').split(" .")
            q = [line.strip(" ") + " ." for line in q if line != ""]
            r = row["r"].strip('"').split(" .")
            r = [line.strip(" ") + " ." for line in r if line != ""]
            if pid not in data:
                data[pid] = {"q": q, "r": r, "s": s, "qq": [], "rr": []}
            if is_test:
                continue
            assert data[pid]["s"] == s and "Same id with different s!!"
            qq = row["q'"].strip('"').split(" .")
            qq = [line.strip(" ") for line in qq if line != ""]
            rr = row["r'"].strip('"').split(" .")
            rr = [line.strip(" ") for line in rr if line != ""]

            data[pid]["qq"] += qq
            data[pid]["rr"] += rr
    return data


def dict_from_raw(data_path: str) -> Dict[str, Dict[str, Any]]:
    """Reads from data_path
    data: Dict[str, Dict[str, Any]]
        = {
            '0': {
                    'q': str,
                    'r': str,
                    's': bool,
                    'qq': List[str],
                    'rr': List[str],
                 },
            ...
           }
    """
    data = {}
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            pid, q, r, s, qq, rr = (
                row["id"],
                row["q"],
                row["r"],
                row["s"],
                row["q'"],
                row["r'"],
            )
            if pid not in data:
                data[pid] = {
                    "q": q,
                    "r": r,
                    "s": s,
                    "qq": [],
                    "rr": [],
                }
            data[pid]["qq"].append(qq)
            data[pid]["rr"].append(rr)
    return data


def tokenize_and_clean(x: str) -> List[str]:
    """Tokenize input string after predifined filtering."""
    symbols = [
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        " ",
        "-",
        # ".",
        "/",
        ":",
        ";",
        "<",
        "=",
        ">",
        # "?",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "{",
        "|",
        "}",
        "~",
        "''",
        "``",
    ]
    x_split = word_tokenize(x)
    x_split = [element for element in x_split if element not in symbols]

    return x_split


def get_word_label(x_split: List[str], xx_split: List[str]) -> List[int]:
    x_labels = []
    xxcount = 0
    for element in x_split:
        if xxcount < len(xx_split) and xx_split[xxcount] == element:
            x_labels += [1]
            xxcount += 1
        else:
            x_labels += [0]

    assert len(x_labels) == len(x_split), "x_labels length mismatch"
    if xxcount != len(xx_split):
        print("xx mismatch:", xx_split[xxcount])

    return x_labels


def data_v3(data_path: str, is_test: bool = False) -> List[Dict[str, Any]]:
    """Construct data for SeqToSeq model with wordwise prediction."""
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        max_length = 0
        data = []
        for row in tqdm(reader):
            pid = row["id"]
            s = row["s"]
            q = row["q"]
            r = row["r"]

            q_split = tokenize_and_clean(q)
            r_split = tokenize_and_clean(r)

            if is_test:
                q_labels = []
                r_labels = []
            else:
                qq = row["q'"]
                rr = row["r'"]
                qq_split = tokenize_and_clean(qq)
                rr_split = tokenize_and_clean(rr)
                q_labels = get_word_label(q_split, qq_split)
                # print(q_split, qq_split, q_labels, sep='\n')
                # exit()
                r_labels = get_word_label(r_split, rr_split)

            data.append(
                {
                    "id": pid,
                    "q": q_split,
                    "r": r_split,
                    "q_labels": q_labels,
                    "r_labels": r_labels,
                    "s": s,
                }
            )

            max_length = max(max_length, max(len(q_split), len(r_split)))
        print("[Stat] max_length: ", max_length)
    return data


def construct_vocab_and_save(
    data: List[Dict[str, Any]],
    data_dir: Path,
) -> None:
    """Build vocab for data_v3"""
    words = Counter()
    for element in data:
        words.update([token.lower() for token in element["q"]])
        words.update([token.lower() for token in element["r"]])

    common_words = {w for w, _ in words.most_common(10000)}
    assert ("agree" in common_words) and ("disagree" in common_words)
    vocab = Vocab(common_words)
    vocab_path = data_dir / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    glove_dim = 0
    glove: Dict[str, List[float]] = {}
    glove_path = data_dir / "glove.840B.300d.txt"
    with open(glove_path, "r", encoding="utf-8") as fp:
        row1 = fp.readline()
        # if the first row is not header
        if not re.match("^[0-9]+ [0-9]+$", row1):
            # seek to 0
            fp.seek(0)
        # otherwise ignore the header

        for line in tqdm(fp):
            cols = line.rstrip().split(" ")
            word = cols[0]
            vector = [float(v) for v in cols[1:]]
            # skip word not in words if words are provided
            if word not in common_words:
                continue
            glove[word] = vector
            glove_dim = len(vector)

    assert all(len(v) == glove_dim for v in glove.values())

    embeddings: List[List[float]] = [
        glove.get(token, [random() * 2 - 1 for _ in range(glove_dim)])
        for token in vocab.tokens
    ]
    embedding_path = data_dir / "embeddings.pt"
    torch.save(torch.tensor(embeddings), str(embedding_path))


if __name__ == "__main__":
    nltk.download("punkt")
    
    # ====================== v3 ======================

    data_to_json = data_v3(f"{DATA_ROOT}/raw.csv", is_test=False)
    with open(f"{DATA_ROOT}/data_v3.json", "w", encoding="utf-8") as fp:
        print("Write back train file....")
        json.dump(data_to_json, fp, indent=4)
        print("Done")

    construct_vocab_and_save(data_to_json, Path(DATA_ROOT))

    data_to_json = data_v3(f"{DATA_ROOT}/test.csv", is_test=True)
    with open(f"{DATA_ROOT}/test_v3.json", "w", encoding="utf-8") as fp:
        print("Write back test file....")
        json.dump(data_to_json, fp, indent=4)
        print("Done")

    # ===================== v2 ======================

    data_to_json = data_v2(f"{DATA_ROOT}/raw.csv", is_test=False)
    with open(f"{DATA_ROOT}/data_v2.json", "w", encoding="utf-8") as fp:
        print("Write back train file....")
        json.dump(data_to_json, fp, indent=4)
        print("Done")

    data_to_json = data_v2(f"{DATA_ROOT}/test.csv", is_test=True)
    with open(f"{DATA_ROOT}/test_v2.json", "w", encoding="utf-8") as fp:
        print("Write back test file....")
        json.dump(data_to_json, fp, indent=4)
        print("Done")