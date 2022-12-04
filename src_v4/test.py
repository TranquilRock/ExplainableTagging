import json
import pickle
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from collections import defaultdict
import csv

from transformers import LongformerTokenizerFast


from utils import set_seed

from dataset import SeqtoSeqDataset

from model import SeqtoSeqModel

def get_args() -> argparse.Namespace:
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    # Random seed
    parser.add_argument("--seed", default=0, type=int)

    # Device
    parser.add_argument(
        "--device",
        type=torch.device,
        help="cpu, cuda, cuda:0, cuda:1",
        default="cuda",
    )

    # Data settings
    parser.add_argument(
        "--data_path", type=Path, default="../../data/data_v3.json")
    parser.add_argument("--cache_dir", type=Path, default="./cache/")
    parser.add_argument("--query_max_length", type=int, default=512)
    parser.add_argument("--document_max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pretrained", type=str,
                        default="allenai/longformer-base-4096")

    # Model settings
    parser.add_argument("--num_classes", type=int, default=2)
    
    # ckpt path
    parser.add_argument(
        "--ckpt_path", default="simple_longformer.ckpt", type=str)

    parser.add_argument(
        "--pred_file", default="submission_long.csv", type=str)
    
    args = parser.parse_args()

    return args


def main(args) -> None:
    set_seed(args.seed)
    device = args.device

    
    # Get data
    with open(args.data_path, newline="", encoding='utf-8') as f:
        data = json.load(f)
        
    tokenizer = LongformerTokenizerFast.from_pretrained(args.pretrained, add_prefix_space=True)
        
    # Set dataset and dataloader
    test_set = SeqtoSeqDataset(
        data,
        tokenizer,
        args.query_max_length,
        args.document_max_length,
        args.num_classes,
        "test")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    model = SeqtoSeqModel(
        args.pretrained,
        args.num_classes
    )
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    model.eval()    

    all_ans = defaultdict()
    with torch.no_grad():
        pid_set = set()
        for pid, split, input_tokens, raw_query in tqdm(test_loader):
            input_tokens = input_tokens.to(device)
            output: torch.Tensor = model(input_tokens)
            output = output[0]
            p = pid[0]
            s = split[0]
            if p not in pid_set:
                ans = defaultdict()
                ans[s] = ""
                all_ans[p] = ans
                pid_set.add(p)
            elif s not in all_ans[p]:
                all_ans[p][s] = ""
            for out, query in zip(output[1:], raw_query):
                if out[1] > out[0]:
                    all_ans[p][s] = all_ans[p][s] + " " + query[0]
            
    with open(args.pred_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'q', 'r'])
        for pid in all_ans.keys():
            q = "\"\"" + all_ans[pid]['q'] + "\"\""
            r = "\"\"" + all_ans[pid]['r'] + "\"\"" 
            writer.writerow([pid, q, r])

    
if __name__ == "__main__":
    main(get_args())
