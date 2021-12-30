import json
import os
from functools import lru_cache
from os.path import join, exists
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd

from src.data.objects.frame import RawFrame, Frame, FrameData
from src.data.objects.stack import Stack


def data_to_stack(stack_dict: Dict[Any, Any]) -> Stack:
    frames = [RawFrame(name=frame["name"], file_name=frame["file_name"], line_num=frame["line_number"],
                       commit_hash=frame["commit_hash"], subsystem=frame["subsystem"])
              for frame in stack_dict["elements"]]
    frames = [Frame(frame, FrameData()) for frame in frames]
    return Stack(stack_dict["id"], int(stack_dict["timestamp"]), frames)


def _dir_stacks_id(dir_path: str, size: int = -1) -> List[int]:
    file_names = os.listdir(dir_path)
    if size > 0:
        file_names = file_names[:size]
    return [int(name[:-5]) for name in file_names]


def _read_supervised(path: str) -> Dict[int, int]:
    df = pd.read_csv(path)
    report_fixers = {}

    for row in df.itertuples():
        report_id = int(row.rid)
        report_fixer = int(row.uid)
        report_fixers[report_id] = report_fixer

    return report_fixers


def assignee_data(reports_dir: str, target_path: str, stack_loader: "StackLoader") \
        -> Tuple[List[int], List[int]]:
    report_fixers = _read_supervised(target_path)
    report_ids = _dir_stacks_id(reports_dir)
    report_id_ts = {report_id: stack_loader(report_id).ts for report_id in report_ids}
    report_id_ts = dict(sorted(report_id_ts.items(), key=lambda kv: kv[1]))
    report_ids = list(report_id_ts.keys())
    fixers = [report_fixers[report_id] for report_id in report_ids]

    return report_ids, fixers


class StackLoader:
    def __init__(self, dir: str):
        self._dir = dir

    @lru_cache(maxsize=300_000)
    def __call__(self, stack_id: int) -> Optional[Stack]:
        stack_path = join(self._dir, f"{stack_id}.json")
        if exists(stack_path):
            with open(stack_path) as file:
                stack_dict = json.load(file)
                return data_to_stack(stack_dict)
