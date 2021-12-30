import numpy as np


def _to_year(ts: int) -> float:
    return ts / 1_000 / 60 / 60 / 24 / 365


def identity_weight_fn(frame_ts: int, user_ts: int) -> float:
    return 1.


def log_weight_fn(frame_ts: int, user_ts: int) -> float:
    return 1 / (np.log(1 + _to_year(frame_ts - user_ts)) + 1)


def exp_weight_fn(frame_ts: int, user_ts: int) -> float:
    return np.exp(-_to_year(frame_ts - user_ts))
