"""Inference for seq2seq"""
import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SeqtoSeqDataset
from model import SeqtoSeqModel
from utils import set_seed


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
        "--data_path", type=Path, default="../../data/test_data_v3.json")
    parser.add_argument("--cache_dir", type=Path, default="./cache/")
    parser.add_argument("--query_max_length", type=int, default=512)
    parser.add_argument("--document_max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)

    # Model settings
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_classes", type=int, default=2)

    # ckpt path
    parser.add_argument(
        "--ckpt_path", default="simple_transformer.ckpt", type=str)
    parser.add_argument(
        "--pred_file", default="submission_simple.csv", type=str)

    args = parser.parse_args()

    return args


def main(args) -> None:
    set_seed(args.seed)
    device = args.device

    # Get Vocab
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    # Get data
    with open(args.data_path, newline="", encoding='utf-8') as f:
        data = json.load(f)

    # Set dataset and dataloader
    test_set = SeqtoSeqDataset(
        data,
        vocab,
        args.query_max_length,
        args.document_max_length,
        args.num_classes,
        "test")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqtoSeqModel(
        embeddings,
        args.d_model,
        args.dim_feedforward,
        args.nhead,
        args.num_layers,
        args.num_classes,
    )
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    model.eval()

    all_ans = defaultdict()
    with torch.no_grad():
        for pid, split, input_tokens, raw_query in tqdm(test_loader):
            input_tokens = input_tokens.to(device)
            output: torch.Tensor = model(input_tokens)
            output = output[0]
            p = pid[0]
            s = split[0]
            for out, query in zip(output, raw_query):
                if out[1] > out[0]:
                    if p not in all_ans:
                        ans = defaultdict()
                        ans[s] = query[0]
                        all_ans[p] = ans
                    else:
                        if s not in all_ans[p]:
                            all_ans[p][s] = query[0]
                        else:
                            all_ans[p][s] = all_ans[p][s] + " " + query[0]

    with open(args.pred_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'q', 'r'])
        for pid in all_ans.keys():
            q = r = "\"\"\"\""
            if 'q' in all_ans[pid].keys():
                q = "\"\"" + all_ans[pid]['q'] + "\"\""
            if 'r' in all_ans[pid].keys():
                r = "\"\"" + all_ans[pid]['r'] + "\"\""
            writer.writerow([pid, q, r])


if __name__ == "__main__":
    main(get_args())
