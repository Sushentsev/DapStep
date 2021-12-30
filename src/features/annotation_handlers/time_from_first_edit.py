from typing import List, Set, Dict

import numpy as np

from src.data.objects.frame import Frame
from src.features.features_base import AnnotationHandler


def _year(milliseconds: int) -> float:
    return milliseconds / 1_000 / 60 / 60 / 24 / 365


class TimeFromFirstEditHandler(AnnotationHandler):
    """
    Time from first modification.
    """

    def __init__(self):
        super().__init__(1, "time_from_first_edit")
        self._user2ts = None
        self._report_ts = None

    def init(self, report_ts: int, frame: Frame):
        self._user2ts = {}
        self._report_ts = report_ts

    def handle_line(self, line_num: int, commit_hash: str, author: int, ts: int):
        if ts > 0:
            self._user2ts[author] = min(self._user2ts.get(author, ts), ts)

    def build(self, user_ids: Set[int]) -> Dict[int, List[float]]:
        return {user_id: [np.log(self._report_ts - self._user2ts[user_id])] for user_id in user_ids}
