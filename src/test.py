"""TODO"""
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange, tqdm
from transformers import (
    RobertaTokenizerFast,
    LongformerTokenizerFast,
    get_linear_schedule_with_warmup,
)
from data import LongformerDatasetV1
from utils import set_seed
import json
from model import LongformerRelationModel

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
    parser.add_argument("--sentence_max_length", default=512, type=int)
    parser.add_argument("--document_max_length", default=2048, type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sentence_pretrained", type=str,
                        default="roberta-base")
    parser.add_argument("--document_pretrained", type=str,
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
    sentence_tokenizer = RobertaTokenizerFast.from_pretrained(
        args.sentence_pretrained)
    document_tokenizer = LongformerTokenizerFast.from_pretrained(
        args.document_pretrained)
    model = LongformerRelationModel(args.sentence_pretrained,
                                    args.document_pretrained, args.num_classes)

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    test_data = copy.deepcopy(data)
    test_set = LongformerDatasetV1(
        test_data, sentence_tokenizer, document_tokenizer, "test",
        args.sentence_max_length, args.document_max_length)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model.eval()

    with torch.no_grad():
        ans = []
        for idx, (test_id, q, r_p, r, q_p, s) in tqdm(enumerate(test_loader)):
            ans_r = ""
            ans_q = ""
            r_p, q_p, s = r_p[0].to(device), q_p[0].to(device), s[0].to(device)
            q = q[0].to(device)
            r = r[0].to(device)

            out_list_0 = []
            for i, q_input in enumerate(q):
                q_input = torch.unsqueeze(q_input, dim=0)
                q_input = q_input.to(device)
                output: torch.Tensor = model(q_input, r_p, s)
                out = output[0]
                if out[0] > out[1]:
                    ans_q = ans_q + " " + data[idx]['q'][i]
                out_list_0.append(out[0].cpu())
            if ans_q == "":
                m_idx = np.argmax(out_list_0)
                ans_q = ans_q + " " + data[idx]['q'][m_idx]

            out_list_0 = []
            for i, r_input in enumerate(r):
                r_input = torch.unsqueeze(r_input, dim=0)
                r_input = r_input.to(device)
                output: torch.Tensor = model(r_input, q_p, s)
                out = output[0]
                if out[0] > out[1]:
                    ans_r = ans_r + " " + data[idx]['r'][i]
                out_list_0.append(out[0].cpu())
            if ans_r == "":
                m_idx = np.argmax(out_list_0)
                ans_r = ans_r + " " + data[idx]['r'][m_idx]

            ans_q = "\"\"" + ans_q + "\"\""
            ans_r = "\"\"" + ans_r + "\"\""
            ans.append([test_id[0].item(), ans_q, ans_r])

    with open(args.pred_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'q', 'r'])
        writer.writerows(ans)


if __name__ == "__main__":
    args = get_args()
    main(args)
