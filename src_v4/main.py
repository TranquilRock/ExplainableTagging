import json
import pickle
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from utils import set_seed

from transformers import AdamW
from transformers import LongformerTokenizerFast

from dataset import SeqtoSeqDataset

from model import SeqtoSeqModel

def get_args() -> argparse.Namespace:
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    # Random seed
    parser.add_argument("--seed", default=1234, type=int)

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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--pretrained", type=str,
                        default="allenai/longformer-base-4096")

    # Training settings
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--logging_step", type=int, default=256)
    parser.add_argument("--gradient_accumulation_step", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)

    # Model settings
    parser.add_argument("--num_classes", type=int, default=2)
    
    # ckpt path
    parser.add_argument(
        "--ckpt_path", default="simple_longformer.ckpt", type=str)
    
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
    train_set = SeqtoSeqDataset(
        data,
        tokenizer,
        args.query_max_length,
        args.document_max_length,
        args.num_classes,
        "train")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = SeqtoSeqModel(
        args.pretrained,
        args.num_classes
    )
    model = model.to(device)

    # Training settings
    logging_step = args.logging_step
    num_epoch = args.num_epoch
    learning_rate = args.lr
    gradient_accumulation_step = args.gradient_accumulation_step
    criterion = torch.nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Start training
    epoch_pbar = trange(num_epoch, desc="Epoch")

    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        for step, (input_tokens, labels) in enumerate(tqdm(train_loader)):
            input_tokens = input_tokens.to(device)
            labels = labels.to(device)
            output: torch.Tensor = model(input_tokens)
            loss = criterion(output[:,1:args.query_max_length+1,:], labels)
            total_loss += loss.item()
            normalized_loss = loss / gradient_accumulation_step
            normalized_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if (step + 1) % gradient_accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % logging_step == 0:
                tqdm.write(
                    f"Epoch: [{epoch}/{num_epoch}], Loss: {total_loss / logging_step:.6f}")
                total_loss = 0
        torch.save(model.state_dict(), args.ckpt_path)
    
if __name__ == "__main__":
    main(get_args())
