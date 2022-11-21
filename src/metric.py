from nltk import tokenize
from typing import List


def lcs_length(s1: List[str], s2: List[str]) -> int:
    """Standard LCS with DP 
    Source: https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
    """
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


if __name__ == "__main__":
    print(cal_score('My guess for qq',
                    'Another guess',
                    [
                        'My guess for qq'
                    ],
                    [
                        'Another guess'
                    ],
                    ))
    print(cal_score('My guess for qq',
                    'Another guess',
                    [
                        "this is line1 in qq.",
                        "Line2 here!!"
                    ],
                    [
                        "C 8 7 6 3",
                        "I want to starburst you.",
                        "Can you guess who I am?",
                        "Another guess:)"
                    ],
                    ))
