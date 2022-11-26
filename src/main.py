import argparse

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from transformers import (
    RobertaTokenizerFast,
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
        "--data_path", default="../data/data_v1.json", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pretrained", type=str,
                        default="roberta-base")

    # Training settings
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--gradient_accumulation_step", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-6)
    
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

    # Load tokenizer and model
    tokenizer = RobertaTokenizerFast.from_pretrained(args.pretrained)
    model = RelationalModel(args.pretrained, args.num_classes)
    model = model.to(device)

    # Prepare Dataset and Dataloader
    train_set = RelationalDataset(
        data, tokenizer, "train", args.max_length)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Training settings
    num_epoch = args.num_epoch
    logging_step = args.logging_step
    learning_rate = args.lr
    gradient_accumulation_step = args.gradient_accumulation_step
    criterion = torch.nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    label1 = torch.concat([torch.ones((args.batch_size, 1)),
                           torch.zeros((args.batch_size, 1))], dim=1).to(device)
    label2 = torch.concat([torch.zeros((args.batch_size, 1)),
                           torch.ones((args.batch_size, 1))], dim=1).to(device)
    # Start training
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    step = 0
    logging_step = 32

    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        for (q, r_p, q_ans, r, q_p, r_ans, s) in tqdm(train_loader):
            r_p, q_p, s = r_p[0].to(device), q_p[0].to(device), s[0].to(device)
            q, q_ans = q[0].to(device), q_ans[0].to(device)
            r, r_ans = r[0].to(device), r_ans[0].to(device)
            
            for q_input, q_label in zip(q, q_ans):
                q_input = torch.unsqueeze(q_input, dim=0)
                output: torch.Tensor = model(q_input, r_p, s)
                loss = criterion(output, label1 if q_label else label2)
                total_loss += loss.item()
                loss.backward()
                step += 1 
                if step % gradient_accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                if step % logging_step == 0:
                    tqdm.write("Epoch: [{}/{}], Loss: {:.6f}".format(epoch,
                                                                     args.num_epoch, total_loss / 2 / logging_step))
                    total_loss = 0

            for r_input, r_label in zip(r, r_ans):
                r_input = torch.unsqueeze(r_input, dim=0)
                output: torch.Tensor = model(r_input, q_p, s)
                loss = criterion(output, label1 if r_label else label2)
                total_loss += loss.item()
                loss.backward()
                step += 1 
                if step % gradient_accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                if step % logging_step == 0:
                    tqdm.write("Epoch: [{}/{}], Loss: {:.6f}".format(epoch,
                                                                     args.num_epoch, total_loss / 2 / logging_step))
                    total_loss = 0
                           
        torch.save(model.state_dict(), "robertaaa.ckpt")

        
        

if __name__ == "__main__":
    args = get_args()
    main(args)
