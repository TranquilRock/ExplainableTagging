"""Sequence to sequence method"""
import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm.auto import tqdm

from data import QDDataset, Vocab
from model import QDNet
from utils import set_seed

ROOT = "/tmp2/b08902011/ExplainableTagging"


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

    dataset = QDDataset(
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

    model = QDNet(
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
        )
    else:
        _test(model, loader, args.ckpt_path, args.pred_path, device)


@torch.no_grad()
def _test(
    model: nn.Module,
    loader: DataLoader,
    ckpt_path: Path,
    pred_path: Path,
    device: torch.device,
) -> None:

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    model.eval()

    all_ans = defaultdict(lambda: defaultdict(list))
    for pid, split, query_tokens, article_tokens, raw_art in tqdm(loader):
        query_tokens = query_tokens.to(device)
        article_tokens = article_tokens.to(device)
        output: torch.Tensor = model(query_tokens, article_tokens).squeeze(
            0
        )  # batch_size == 1
        pid: str = pid[0]
        split: str = split[0]
        for out, query in zip(output, raw_art):
            if out.argmax() == 1:
                all_ans[pid][split].append(query[0])

    with open(pred_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "q", "r"])
        for pid, ans in all_ans.items():
            q = f'""{" ".join(ans["q"])}""' if "q" in ans else '""""'
            r = f'""{" ".join(ans["r"])}""' if "r" in ans else '""""'
            writer.writerow([pid, q, r])


def _train(
    model: nn.Module,
    loader: DataLoader,
    lr: float,
    num_epoch: int,
    accumulation: int,
    ckpt_path: Path,
    device: torch.device,
) -> None:
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.02, 0.98]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_pbar = trange(num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        progress = tqdm(loader)
        for step, (query_tokens, article_tokens, labels) in enumerate(progress):
            query_tokens = query_tokens.to(device)
            article_tokens = article_tokens.to(device)
            labels = labels.view(-1).to(device)
            output: torch.Tensor = model(query_tokens, article_tokens)
            loss = criterion(
                output[:, : labels.size(0), :].contiguous().view(-1, 2),
                labels,
            )
            total_loss += loss.item()
            normalized_loss = loss / accumulation
            normalized_loss.backward()
            clip_grad_norm_(model.parameters(), 1)
            if (step + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            progress.set_description(
                f"Epoch: [{epoch}/{num_epoch}], Loss: {total_loss / (step + 1):.6f}"
            )
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0xC8763, type=int)
    parser.add_argument(
        "--device",
        type=torch.device,
        help="cpu, cuda, cuda:0, cuda:1",
        default="cuda",
    )

    # Data settings
    parser.add_argument("--data_path", type=Path, default=f"{ROOT}/data/data_v3.json")
    parser.add_argument("--cache_dir", type=Path, default=f"{ROOT}/data")
    parser.add_argument("--query_max_length", type=int, default=1024)
    parser.add_argument("--document_max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)

    # Training settings
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--accumulation", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Model settings
    parser.add_argument("--d_model", type=int, default=400)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)

    # ckpt path
    parser.add_argument("--ckpt_path", default=f"{ROOT}/ckpt/qd.ckpt", type=str)
    parser.add_argument("--pred_path", default=f"{ROOT}/pred/out.csv", type=str)
    parser.add_argument("--train", action="store_true")

    main(parser.parse_args())
