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

import copy
import csv

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
        "--data_path", default="../data/test_data.json", type=str)
    parser.add_argument("--max_length", default=2048, type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pretrained", type=str,
                        default="allenai/longformer-base-4096")

    # Train valid split
    parser.add_argument("--valid_size", default=0.2, type=float)

    # Model settings
    parser.add_argument("--num_classes", type=int, default=2)

    # ckpt path
    parser.add_argument(
        "--ckpt_path", default="tmp.ckpt", type=str)

    parser.add_argument(
        "--pred_file", default="submission_0.csv", type=str)

    args = parser.parse_args()

    return args


def main(args) -> None:
    set_seed(args.seed)
    device = args.device

    # Get data
    with open(args.data_path, newline="") as f:
        data = json.load(f)

    # Load tokenizer and model
    tokenizer = LongformerTokenizerFast.from_pretrained(args.pretrained)
    model = RelationalModel(args.pretrained, args.num_classes)

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    test_data = copy.deepcopy(data)
    test_set = RelationalDataset(test_data, tokenizer, "test", args.max_length)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    model.eval()
    
    with torch.no_grad():
        ans = []
        for idx, (test_id, q_r_seqs, r_q_seqs) in tqdm(enumerate(test_loader)):
            ans_r = ""
            ans_q = ""
            for i, (q_input, r_input)  in enumerate(zip(q_r_seqs, r_q_seqs)):
                q_input, r_input = q_input.to(device), r_input.to(device)
                output: torch.Tensor = model(q_input)
                out = output[0]
                if out[1] > out[0]:
                    ans_r = ans_r + " " + data[idx]['r'][i]
                  
                output: torch.Tensor = model(r_input)
                out = output[0]
                if out[1] > out[0]:
                    ans_q = ans_q + " " +  data[idx]['q'][i]

            ans_q = "\"\"" + ans_q +  "\"\""
            ans_r = "\"\"" + ans_r +  "\"\""
            ans.append([test_id[0].item(), ans_q, ans_r])
            
    with open(args.pred_file, 'w') as fp:
         writer = csv.writer(fp)
         writer.writerow(['id', 'q', 'r'])
         writer.writerows(ans)             
        
if __name__ == "__main__":
    args = get_args()
    main(args)
