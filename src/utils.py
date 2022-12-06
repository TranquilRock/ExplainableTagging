"""Helper functions."""
import random
from typing import List

import torch
from nltk import tokenize


def lcs_length(str_1: List[str], str_2: List[str]) -> int:
    """Standard LCS with DP
    Source: https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
    """
    if len(str_2) > len(str_1):
        str_1, str_2 = str_2, str_1
    lcs = [[0] * (len(str_2) + 1) for _ in range(2)]
    for i in range(1, len(str_1)+1):
        for j in range(1, len(str_2)+1):
            if str_1[i-1] == str_2[j-1]:
                lcs[i % 2][j] = lcs[(i-1) % 2][j-1] + 1
            else:
                lcs[i % 2][j] = max(lcs[(i-1) % 2][j], lcs[i % 2][j-1])
    return lcs[len(str_1) % 2][len(str_2)]


def cal_score(q_rel_guess: str, r_rel_guess: str, q_rel: List[str], r_rel: List[str]) -> float:
    """Calculate the score per id."""

    tokenized_qq_guess = tokenize.word_tokenize(q_rel_guess)
    tokenized_rr_guess = tokenize.word_tokenize(r_rel_guess)

    tokenized_qq = [tokenize.word_tokenize(line) for line in q_rel]
    tokenized_rr = [tokenize.word_tokenize(line) for line in r_rel]

    ret = float('-inf')
    for qq_ans, rr_ans in zip(tokenized_qq, tokenized_rr):
        q_lcs = lcs_length(tokenized_qq_guess, qq_ans)
        r_lcs = lcs_length(tokenized_rr_guess, rr_ans)
        ret = max(ret,
                  (q_lcs) / (len(qq_ans) + len(tokenized_qq_guess) - q_lcs) +
                  (r_lcs) / (len(rr_ans) + len(tokenized_rr_guess) - r_lcs))
    return ret


def set_seed(seed: int) -> None:
    """
    Fixed seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True # type: ignore

def concat_child(answer_list: List[List[str]]) -> List[List[str]]:
    """
    Originally the structure looks like
        [
            ["Answer line1", "Answer line2", ...]
        ]
    Transform into:
        [
            ["Answer", "line1", "Answer, "line2", ...]
        ]
    For LCS matching.
    """
    for i in range(len(answer_list)):
        tmp_entry = []
        for line in answer_list[i]:
            tmp_entry += line.split(' ')
        answer_list[i] = tmp_entry
    return answer_list
