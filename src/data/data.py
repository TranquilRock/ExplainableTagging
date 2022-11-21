from typing import List, Dict, Union, Any
import json

import csv


def json_from_raw(data_path: str = "../../data/data.csv") -> List[Dict[str, Union[int, bool, List[str]]]]:
    """Reads raw data file from data_path
    data = [{
            'id': 8,
            'q': ['It can go both ways .', 'We all doubt .', 'It is what you do with it that matters .'],
            'r': ['True .'],
            's': True,
            'qq': ['It can go both ways', 'We all doubt', 'It is what you do with it that matters'],
            'rr': ['True'],
            }, ]
    """
    data = []
    with open(data_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            id = int(row["id"])
            s = row["s"] == 'AGREE'
            q = row["q"].strip('"').split('.')
            q = [line.strip(' ') + ' .' for line in q if line != '']
            r = row["r"].strip('"').split('.')
            r = [line.strip(' ') + ' .' for line in r if line != '']
            qq = row["q'"].strip('"').split('.')
            qq = [line.strip(' ') for line in qq if line != '']
            rr = row["r'"].strip('"').split('.')
            rr = [line.strip(' ') for line in rr if line != '']
            data.append({
                "id": id,
                "q": q,
                "r": r,
                "s": s,
                "qq": qq,
                "rr": rr,
            })
    return data


def dict_from_raw(data_path: str = "../../data/data.csv") -> Dict[str, Dict[str, Any]]:
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


if __name__ == "__main__":
    with open("./data_v1.json", "w") as f:
        json.dump(json_from_raw(), f, indent=4)
