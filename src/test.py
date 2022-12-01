"""TEST"""
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange, tqdm
from transformers import LongformerTokenizerFast
import json
from data import LongformerDataset
from model import LongformerRelationModel
from utils import set_seed

from collections import defaultdict

import copy
import csv


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
        "--data_path", default="../../data/test_v2.json", type=str)
    parser.add_argument("--sentence_max_length", default=512, type=int)
    parser.add_argument("--document_max_length", default=2048, type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pretrained", type=str,
                        default="allenai/longformer-base-4096")
    
    # Model settings
    parser.add_argument("--num_classes", type=int, default=2)

    # ckpt path
    parser.add_argument(
        "--ckpt_path", default="robertalong.ckpt", type=str)

    parser.add_argument(
        "--pred_file", default="submission_final.csv", type=str)

    args = parser.parse_args()

    return args


def main(args) -> None:
    set_seed(args.seed)
    device = args.device

    # Get data
    with open(args.data_path, newline="") as f:
        data = json.load(f)

    # Load tokenizer and model
    # Load tokenizer and model
    tokenizer = LongformerTokenizerFast.from_pretrained(args.pretrained)
    model = LongformerRelationModel(args.pretrained, args.num_classes)

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    test_set = LongformerDataset(
        data, tokenizer, "test", args.sentence_max_length, args.document_max_length)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model.eval()    

    all_ans = defaultdict()
    with torch.no_grad():
        for pids, splits, inputs, sentences in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs: torch.Tensor = model(inputs)
            for pid, split, output, sentence in zip(pids, splits, outputs, sentences):
                if output[0] > output[1]:
                    if pid not in all_ans:
                        ans = defaultdict()
                        ans[split] = sentence
                        all_ans[pid] = ans
                    else:
                        if split not in all_ans[pid]:
                            all_ans[pid][split] = sentence
                        else:
                            all_ans[pid][split] = all_ans[pid][split] + " " + sentence
                            
    with open(args.pred_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'q', 'r'])
        for pid in all_ans.keys():
            q = "\"\"" + all_ans[pid]['q'] + "\"\""
            r = "\"\"" + all_ans[pid]['r'] + "\"\"" 
            writer.writerow([pid, q, r])

            
if __name__ == "__main__":
    args = get_args()
    main(args)
