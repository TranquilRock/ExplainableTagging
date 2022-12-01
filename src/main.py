"""Main"""
import argparse
import json

import torch
import torch.optim as optim
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import LongformerTokenizerFast

from data import LongformerDataset
from model import LongformerRelationModel
from utils import set_seed

# from transformers import get_linear_schedule_with_warmup


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
        "--data_path", default="../../data/data_v2.json", type=str)
    parser.add_argument("--sentence_max_length", default=256, type=int)
    parser.add_argument("--document_max_length", default=1024, type=int)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--pretrained", type=str,
                        default="allenai/longformer-base-4096")

    # Training settings
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--logging_step", type=int, default=256)
    parser.add_argument("--gradient_accumulation_step", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)

    # Model settings
    parser.add_argument("--num_classes", type=int, default=2)

    args = parser.parse_args()

    return args


def main(args) -> None:
    """Experimental dataset on LongformerDatasetV2"""
    set_seed(args.seed)
    device = args.device

    # Get data
    with open(args.data_path, newline="", encoding='utf-8') as f:
        data = json.load(f)

    # Load tokenizer and model
    tokenizer = LongformerTokenizerFast.from_pretrained(args.pretrained)
    model = LongformerRelationModel(args.pretrained, args.num_classes)
    model = model.to(device)

    # Prepare Dataset and Dataloader
    train_set = LongformerDataset(
        data, tokenizer, "train", args.sentence_max_length, args.document_max_length)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Training settings
    logging_step = args.logging_step
    num_epoch = args.num_epoch
    learning_rate = args.lr
    gradient_accumulation_step = args.gradient_accumulation_step
    criterion = torch.nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Start training
    epoch_pbar = trange(num_epoch, desc="Epoch")

    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        for step, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            output: torch.Tensor = model(inputs)
            loss = criterion(output, labels)
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
        torch.save(model.state_dict(), "robertalong.ckpt")

        
if __name__ == "__main__":
    main(get_args())
