from typing import List, Callable, Iterable, Tuple

import numpy as np


def acc_top_k(y_true: List[int], y_pred: List[List[float]], k_values: List[int]) -> List[float]:
    return [acc_k(y_true, y_pred, k) for k in k_values]


def acc_k(y_true: List[int], y_pred: List[List[float]], k: int = 1) -> float:
    ok_cnt = 0
    for true_user_id, scores in zip(y_true, y_pred):
        indices = (-np.array(scores)).argsort()[:k]
        if true_user_id in indices:
            ok_cnt += 1

    return ok_cnt / len(y_true)


def mrr(y_true: List[int], y_pred: List[List[float]]) -> float:
    rec_ranks = []
    for true_user_id, scores in zip(y_true, y_pred):
        if true_user_id < 0:
            rec_ranks.append(0)
            continue

        desc_ind = list(np.argsort(scores))[::-1]
        rank = desc_ind.index(true_user_id) + 1
        rec_ranks.append(1 / rank)

    return sum(rec_ranks) / len(y_true)


def bootstrap_metric(y_true: List[int], y_pred: List[List[float]],
                     metric: Callable[[Iterable, Iterable], float],
                     err: float = 0.05, iters: int = 100, size: float = 1.) -> Tuple[float, float, float]:
    values = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    real_value = metric(y_true, y_pred)
    n = len(y_true)
    sn = int(size * n)
    left = int(iters * err / 2)
    while len(values) < iters:
        inds = np.random.choice(n, sn)
        value = metric(y_true[inds], y_pred[inds])
        values.append(value)
    values = sorted(values)
    return round(real_value, 4), round(values[left], 4), round(values[iters - 1 - left], 4)


def boostrap_comp_metric(y_true: List[int], y_pred_1: List[List[float]], y_pred_2: List[List[float]],
                         metric: Callable[[Iterable, Iterable], float],
                         err: float = 0.05, iters: int = 100, size: float = 1.) -> Tuple[float, float]:
    values = []
    y_true = np.array(y_true)
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)
    n = len(y_true)
    sn = int(size * n)
    left = int(iters * err / 2)
    while len(values) < iters:
        inds = np.random.choice(n, sn)
        value_1 = metric(y_true[inds], y_pred_1[inds])
        value_2 = metric(y_true[inds], y_pred_2[inds])
        values.append(value_1 - value_2)
    values = sorted(values)
    return round(values[left], 4), round(values[iters - 1 - left], 4)
