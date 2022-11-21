import itertools
from collections import defaultdict
from typing import List


def lcs_length(s1: List[str], s2: List[str]) -> int:
    """Standard LCS with DP from https://en.wikipedia.org/wiki/Longest_common_subsequence_problem"""
    if len(s2) > len(s1):
        s1, s2 = s2, s1
    lcs = [[0] * (len(s2) + 1) for _ in range(2)]
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:
                lcs[i % 2][j] = lcs[(i-1) % 2][j-1] + 1
            else:
                lcs[i % 2][j] = max(lcs[(i-1) % 2][j], lcs[i % 2][j-1])
    return lcs[len(s1) % 2][len(s2)]


def _concat_child(answer_list: List[List[str]]) -> List[List[str]]:
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
        qq[i] = tmp_entry
    return answer_list


def cal_score(qq_guess: List[str], rr_guess: List[str], qq: List[List[str]], rr: List[List[str]]) -> float:
    """Calculate the score per id."""
    assert len(qq) == len(rr) and "Length of qq and rr mismatch!!"
    qq = _concat_child(qq)
    rr = _concat_child(rr)
    ret = float('-inf')
    for qq_ans, rr_ans in zip(qq, rr):
        q_lcs = lcs_length(qq_guess, qq_ans)
        r_lcs = lcs_length(rr_guess, rr_ans)
        ret = max((q_lcs) / (len(qq_ans) + len(qq_guess) - q_lcs) +
                  (r_lcs) / (len(rr_ans) + len(rr_guess) - r_lcs))
    return ret


if __name__ == "__main__":
    cal_score('', '', [["1 2 3", "5 6 7"], ["c 8 7 6 3", "www www"], ], [['']])
