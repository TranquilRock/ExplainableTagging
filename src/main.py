import argparse

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from transformers import (
    LongformerTokenizerFast,
    get_linear_schedule_with_warmup,
)
from data import RelationalDataset
from utils import set_seed
import json
from model import RelationalModel


def get_args() -> argparse.Namespace:
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    # Random seed
    parser.add_argument("--seed", default=0xC8763, type=int)

    # Device
    parser.add_argument(
        "--device",
        type=torch.device,
        help="cpu, cuda, cuda:0, cuda:1",
        default="cuda",
    )

    # Data settings
    parser.add_argument(
        "--data_path", default="../data/data_v1.json", type=str)
    parser.add_argument("--max_length", default=2048, type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pretrained", type=str,
                        default="allenai/longformer-base-4096")

    # Train valid split
    parser.add_argument("--valid_size", default=0.2, type=float)

    # Training settings
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--gradient_accumulation_step", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)

    # Model settings
    parser.add_argument("--num_classes", type=int, default=2)

    args = parser.parse_args()

    return args


def main(args) -> None:
    set_seed(args.seed)
    device = args.device

    # Get data
    with open(args.data_path, newline="") as f:
        data = json.load(f)

    # Split train valid ids
    train_ids = list(set([dic['id'] for dic in data]))
    train_ids, _ = train_test_split(
        train_ids, test_size=args.valid_size)
    train_ids = set(train_ids)

    train_data, dev_data = [], []

    for entry in data:
        if entry['id'] in train_ids:
            train_data.append(entry)
        else:
            dev_data.append(entry)
    del train_ids
    del data

    # Load tokenizer and model
    tokenizer = LongformerTokenizerFast.from_pretrained(args.pretrained)
    model = RelationalModel(args.pretrained, args.num_classes)
    model = model.to(device)

    # Prepare Dataset and Dataloader
    train_set = RelationalDataset(
        train_data, tokenizer, "train", args.max_length)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    dev_set = RelationalDataset(dev_data, tokenizer, "dev", args.max_length)
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Training settings
    num_epoch = args.num_epoch
    logging_step = args.logging_step
    learning_rate = args.lr
    gradient_accumulation_step = args.gradient_accumulation_step
    criterion = torch.nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    '''
    update_step = (
        num_epoch * len(train_loader) // gradient_accumulation_step + num_epoch
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0.1 * update_step, update_step
    )
    '''
    label1 = torch.concat([torch.ones((args.batch_size, 1)),
                           torch.zeros((args.batch_size, 1))], dim=1).to(device)
    label2 = torch.concat([torch.zeros((args.batch_size, 1)),
                           torch.ones((args.batch_size, 1))], dim=1).to(device)
    # Start training
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    #step = 0

    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0
        for step, (q_r_seqs, q_ans, r_q_seqs, r_ans) in enumerate(train_loader):
            for q_input, q_label, r_input, r_label in zip(q_r_seqs, q_ans, r_q_seqs, r_ans):
                q_input, q_label, r_input, r_label = q_input.to(device), q_label.to(
                    device), r_input.to(device), r_label.to(device)
                optimizer.zero_grad()
                output: torch.Tensor = model(q_input)
                loss = criterion(output, label1 if q_label else label2)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                output: torch.Tensor = model(r_input)
                loss = criterion(output, label1 if r_label else label2)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            tqdm.write("Epoch: [{}/{}], Loss: {:.6f}".format(epoch,
                       args.num_epoch, total_loss / 2 / (step + 1)))
        torch.save(model.state_dict(), "/tmp2/b08902011/ckpt/tmp.ckpt")

        '''
        if step % gradient_accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        '''


if __name__ == "__main__":
    args = get_args()
    main(args)
