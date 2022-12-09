import json
import logging
import pickle
import re
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from random import random, seed
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from utils import set_seed
from utils import Vocab

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def build_vocab(
    words: Counter, vocab_size: int, output_dir: Path, glove_path: Path
) -> None:
    common_words = {w for w, _ in words.most_common(vocab_size)}
    assert (('agree' in common_words) and ('disagree' in common_words))
    vocab = Vocab(common_words)
    vocab_path = output_dir / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    logging.info(f"Vocab saved at {str(vocab_path.resolve())}")

    glove: Dict[str, List[float]] = {}
    logging.info(f"Loading glove: {str(glove_path.resolve())}")
    with open(glove_path) as fp:
        row1 = fp.readline()
        # if the first row is not header
        if not re.match("^[0-9]+ [0-9]+$", row1):
            # seek to 0
            fp.seek(0)
        # otherwise ignore the header

        for i, line in tqdm(enumerate(fp)):
            cols = line.rstrip().split(" ")
            word = cols[0]
            vector = [float(v) for v in cols[1:]]

            # skip word not in words if words are provided
            if word not in common_words:
                continue
            glove[word] = vector
            glove_dim = len(vector)

    assert all(len(v) == glove_dim for v in glove.values())
    assert len(glove) <= vocab_size

    num_matched = sum([token in glove for token in vocab.tokens])
    logging.info(
        f"Token covered: {num_matched} / {len(vocab.tokens)} = {num_matched / len(vocab.tokens)}"
    )
    embeddings: List[List[float]] = [
        glove.get(token, [random() * 2 - 1 for _ in range(glove_dim)])
        for token in vocab.tokens
    ]
    embeddings = torch.tensor(embeddings)
    embedding_path = output_dir / "embeddings.pt"
    torch.save(embeddings, str(embedding_path))
    logging.info(f"Embedding shape: {embeddings.shape}")
    logging.info(f"Embedding saved at {str(embedding_path.resolve())}")


def main(args):
    set_seed(args.seed)

    words = Counter()
    with open(args.data_path, newline="", encoding='utf-8') as f:
        data = json.load(f)

    for element in data:
        words.update([token.lower() for token in element['q']])
        words.update([token.lower() for token in element['r']])

    build_vocab(words, args.vocab_size, args.output_dir, args.glove_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Directory to the dataset.",
        default="../../data/data_last.json",
    )
    parser.add_argument(
        "--glove_path",
        type=Path,
        help="Path to Glove Embedding.",
        default="./glove.840B.300d.txt",
    )
    parser.add_argument("--seed", type=int, help="Random seed.", default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Number of token in the vocabulary",
        default=10_000,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
