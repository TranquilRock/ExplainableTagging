import csv
import json
from typing import Any, Dict, List, Union

from nltk.tokenize import word_tokenize
from tqdm import tqdm


def get_split(x):
    symbols = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':',
                       ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', "''", "``", "--"]
    x_split = word_tokenize(x)
    x_split = [element for element in x_split if element not in symbols]

    return x_split


def get_labels(x_split, xx_split):
    x_labels = []
    xxcount = 0
    for element in x_split:
        if xxcount == len(xx_split):
            x_labels += [0]
        elif xx_split[xxcount] == element:
            x_labels += [1]
            xxcount += 1
        else:
            x_labels += [0]
                    
    assert len(x_labels) == len(x_split), "x_labels length mismatch"
    if xxcount != len(xx_split):
        print("xx mismatch:", xx_split[xxcount])
        
    return x_labels


def data_v3(data_path: str, mode: str):
    with open(data_path, newline="", encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=",")
        max_length = 0
        data = []
        for row in tqdm(reader):
            pid = row["id"]
            s = row["s"]
            q = row["q"]
            r = row["r"]

            q_split = get_split(q)
            r_split = get_split(r)

            if mode == "train":
                qq = row["q'"]
                rr = row["r'"]
                qq_split = get_split(qq)
                rr_split = get_split(rr)
                q_labels = get_labels(q_split, qq_split)
                r_labels = get_labels(r_split, rr_split)
            else:
                q_labels = []
                r_labels = []
                
            data.append({
                "id": pid,
                "q": q_split,
                "r": r_split,
                "q_labels": q_labels,
                "r_labels": r_labels,
                "s": s
            })
            
            max_length = max(max_length, max(len(q_split), len(r_split)))
        print("[Stat] max_length: ", max_length)
    
    return data
            

if __name__ == "__main__":
    DATA_ROOT = "/tmp2/b08902123/data/"
    data_path = f"{DATA_ROOT}/test_data.csv"
    dest_path = f"{DATA_ROOT}/test_data_last.json"
    mode = "test"
    data_to_json = data_v3(data_path, mode)
    with open(dest_path, "w", encoding='utf-8') as f:
        print("Write back....")
        json.dump(data_to_json, f, indent=4)
        print("Done")
