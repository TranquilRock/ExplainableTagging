import argparse
import csv
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import SeqtoSeqDataset, Vocab
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from model import SeqtoSeqModel
from utils import set_seed

REPO_ROOT = "/tmp2/b08902011/ExplainableTagging"


def main(args) -> None:
    """Main entry."""
    set_seed(args.seed)
    device = args.device

    # Get Vocab
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    # Get data
    with open(args.data_path, newline="", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    dataset = SeqtoSeqDataset(
        data,
        vocab,
        args.query_max_length,
        args.document_max_length,
        "train" if args.train else "test",
    )

    loader = DataLoader(
        dataset,
        args.batch_size if args.train else 1,
        shuffle=True,
        num_workers=8,
    )

    model = SeqtoSeqModel(
        embeddings,
        args.d_model,
        args.dim_feedforward,
        args.nhead,
        args.num_layers,
        args.dropout,
        num_classes=2,
    )

    model = model.to(device)

    if args.train:
        _train(
            model,
            loader,
            args.lr,
            args.num_epoch,
            args.accumulation,
            args.ckpt_path,
            device,
            args.query_max_length,
            args.document_max_length,
        )
    else:
        _test(
            model,
            loader,
            args.ckpt_path,
            args.pred_path,
            device,
            args.query_max_length,
        )


def _train(
    model: nn.Module,
    loader: DataLoader,
    lr: float,
    num_epoch: int,
    accumulation: int,
    ckpt_path: Path,
    device: torch.device,
    query_max_length: int,
    document_max_length: int,
) -> None:

    # Training settings
    logging_step = 1
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Start training
    epoch_pbar = trange(num_epoch, desc="Epoch")

    weights = [0.5, 0.5]
    nonzeros_num = 0
    zeros_num = 0
    all_num = 0
    for (input_tokens, labels) in loader:
        alln = labels.shape[0] * labels.shape[1]
        nonzeros = torch.count_nonzero(labels)
        zeros = alln - nonzeros
        all_num += alln
        nonzeros_num += nonzeros
        zeros_num += zeros

    weights = torch.tensor([20 * nonzeros_num / all_num, zeros_num / all_num]).to(
        device
    )

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        for step, (input_tokens, labels) in enumerate(tqdm(loader)):
            input_tokens = input_tokens.to(device)
            labels = labels.to(device)
            output: torch.Tensor = model(input_tokens)
            output = output[:, : query_max_length + document_max_length, :].contiguous()
            loss = criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            total_loss += loss.item()
            normalized_loss = loss / accumulation
            normalized_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if (step + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % logging_step == 0:
                tqdm.write(
                    f"Epoch: [{epoch}/{num_epoch}], Loss: {total_loss / logging_step:.6f}"
                )
                total_loss = 0
        torch.save(model.state_dict(), ckpt_path)
        torch.save(model.embed, "embed_4.ckpt")


def _test(
    model: nn.Module,
    loader: DataLoader,
    ckpt_path: Path,
    pred_path: Path,
    device: torch.device,
    query_max_length: int,
) -> None:

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    all_ans = []
    with torch.no_grad():
        for pid, input_tokens, raw_query, raw_document in tqdm(loader):
            input_tokens = input_tokens.to(device)
            output: torch.Tensor = model(input_tokens)
            output = output[0]
            p = pid[0]
            ans_q = ""
            ans_r = ""
            for out, query in zip(output, raw_query):
                if out[1] > out[0]:
                    ans_q = ans_q + " " + query[0]
            for out, query in zip(output[query_max_length:], raw_document):
                if out[1] > out[0]:
                    ans_r = ans_r + " " + query[0]
            all_ans.append([p, ans_q, ans_r])

    with open(pred_path, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "q", "r"])
        for d in all_ans:
            pid = d[0]
            q = '""' + d[1] + '""'
            r = '""' + d[2] + '""'
            writer.writerow([pid, q, r])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument(
        "--device",
        type=torch.device,
        help="cpu, cuda, cuda:0, cuda:1",
        default="cuda",
    )

    # Data settings
    parser.add_argument("--data_path", type=Path, default=f"{REPO_ROOT}/data/data_v3.json")
    parser.add_argument("--cache_dir", type=Path, default=f"{REPO_ROOT}/data")
    parser.add_argument("--query_max_length", type=int, default=1024)
    parser.add_argument("--document_max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)

    # Training settings
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--logging_step", type=int, default=4096)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)

    # Model settings
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ckpt path
    parser.add_argument(
        "--ckpt_path", default=f"{REPO_ROOT}/ckpt/simple_transformer_4.ckpt", type=str
    )
    parser.add_argument(
        "--pred_file", default=f"{REPO_ROOT}/pred/submission_better_4.csv", type=str
    )
    arguments = parser.parse_args()
    main(arguments)
