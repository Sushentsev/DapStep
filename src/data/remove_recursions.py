import argparse
import json
import os
from collections import namedtuple
from copy import deepcopy
from typing import List, Dict, Any

from tqdm import tqdm

Cycle = namedtuple("Cycle", ["rep", "period", "pos"])


def _find_dominant(seq: List[str]) -> Cycle:
    best = Cycle(0, 0, 0)  # repetitions - 1, period, position
    period = 0

    while period < len(seq) // max(2, 1 + best.rep):
        period += 1
        length = 0

        for pos in range(len(seq) - 1 - period, -1, -1):
            if seq[pos] == seq[pos + period]:
                length += 1
                repetitions = length // period
                if repetitions >= best.rep:
                    best = Cycle(repetitions, period, pos)
            else:
                length = 0

    return best


def remove_recursion(report: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Remove all recursion from stack trace.
    The algorithm as follows:
    1. Find best cycle
    2. Replace loop with first occurrence
    3. Repeat steps 1-2 until all cycles are deleted

    :param report: bug report in JSON format
    :return: bug report without cycles
    """

    found_cycle = True
    upd_report = deepcopy(report)

    while found_cycle:
        rev_frames = upd_report["elements"][::-1]
        # Find cycle in {method_name}_{line_num}
        tokens = [frame["name"] + "_" + str(frame["line_number"] if frame["line_number"] is not None else -1)
                  for frame in rev_frames]

        cycle = _find_dominant(tokens)

        if cycle.rep > 0:
            found_cycle = True
            rev_upd_frames = [frame for i, frame in enumerate(rev_frames)
                              if i < cycle.pos + cycle.period or i >= cycle.pos + (cycle.rep + 1) * cycle.period]
        else:
            found_cycle = False
            rev_upd_frames = rev_frames

        upd_report["elements"] = rev_upd_frames[::-1]

    return upd_report


def main(reports_dir: str, save_dir: str):
    for report_name in tqdm(os.listdir(reports_dir)):
        with open(f"{reports_dir}/{report_name}") as file:
            report = json.load(file)

        upd_report = remove_recursion(report)
        with open(f"{save_dir}/{report_name}", "w+") as file:
            json.dump(upd_report, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    main(args.reports_dir, args.save_dir)
