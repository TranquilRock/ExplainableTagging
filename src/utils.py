"""Helper functions."""
import random
from typing import Iterable, List, Optional

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


def cal_score(qq_guess: str, rr_guess: str, qq: List[str], rr: List[str]) -> float:
    """Calculate the score per id."""

    qq_guess = tokenize.word_tokenize(qq_guess)
    rr_guess = tokenize.word_tokenize(rr_guess)

    qq = [tokenize.word_tokenize(entry) for entry in qq]
    rr = [tokenize.word_tokenize(entry) for entry in rr]

    ret = float('-inf')
    for qq_ans, rr_ans in zip(qq, rr):
        q_lcs = lcs_length(qq_guess, qq_ans)
        r_lcs = lcs_length(rr_guess, rr_ans)
        ret = max(ret,
                  (q_lcs) / (len(qq_ans) + len(qq_guess) - q_lcs) +
                  (r_lcs) / (len(rr_ans) + len(rr_guess) - r_lcs))
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    L = len(answer_list)
    for i in range(L):
        tmp_entry = []
        for line in answer_list[i]:
            tmp_entry += line.split(' ')
        answer_list[i] = tmp_entry
    return answer_list


