import argparse

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import (
    LongformerTokenizerFast,
    LongformerForQuestionAnswering,
    BertForQuestionAnswering,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from data.dataset import RelationalDataset
from utils import get_data, set_seed


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
        default="cuda:4",
    )

    # Data settings
    parser.add_argument("--data_path", default="./data.csv", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--pretrained", type=str, default="bert-base-cased")

    # Train valid split
    parser.add_argument("--valid_size", default=0.2, type=float)

    # Training settings
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--gradient_accumulation_step", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)

    args = parser.parse_args()

    return args


def get_tokenizer_and_model(name: str):
    if "bert" in name:
        tokenizer = BertTokenizerFast.from_pretrained(name)
        model = BertForQuestionAnswering.from_pretrained(name)
    elif "longformer" in name:
        tokenizer = LongformerTokenizerFast.from_pretrained(name)
        model = LongformerForQuestionAnswering.from_pretrained(name)
    return tokenizer, model


def main(args) -> None:
    set_seed(args.seed)
    device = args.device

    # Get data
    tokenizer, model = get_tokenizer_and_model(args.pretrained)
    data = get_data(args.data_path, tokenizer, args.max_length)

    # Split train valid ids
    train_ids = list(data.keys())
    train_ids, valid_ids = train_test_split(
        train_ids, test_size=args.valid_size)

    # Prepare Dataset and Dataloader
    trainset = RelationalDataset(data, train_ids, "train")
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True)
    print(trainset.get_max_data()[0])
    exit()

    validset = RelationalDataset(data, valid_ids, "valid")
    valid_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = model.to(device)

    # Training settings
    num_epoch = args.num_epoch
    logging_step = args.logging_step
    learning_rate = args.lr
    gradient_accumulation_step = args.gradient_accumulation_step

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    update_step = (
        num_epoch * len(train_loader) // gradient_accumulation_step + num_epoch
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0.1 * update_step, update_step
    )

    # Start training
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        for ID, q, r, q_plum, r_plum in train_loader:
            print(ID, q, r, q_plum, r_plum)
            break
        scheduler.step()


if __name__ == "__main__":
    args = get_args()
    main(args)