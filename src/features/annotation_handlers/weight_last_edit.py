from collections import defaultdict
from typing import List, Set, Dict

import numpy as np

from src.data.objects.frame import Frame
from src.features.features_base import AnnotationHandler
from src.models.weight_fns import exp_weight_fn


class WeightLastEditHandler(AnnotationHandler):
    """
    Weight of last edit.
    """

    def __init__(self):
        super().__init__(2, "weight_last_edit")
        self._report_ts = None
        self._weight_fn = exp_weight_fn  # (report_ts, last_user_ts) -> weight
        self._user2ts = None

    def init(self, report_ts: int, frame: Frame):
        self._report_ts = report_ts
        self._user2ts = defaultdict(lambda: 0)

    def handle_line(self, line_num: int, commit_hash: str, author: int, ts: int):
        self._user2ts[author] = max(self._user2ts[author], ts)

    def build(self, user_ids: Set[int]) -> Dict[int, List[float]]:
        return {user_id: [self._weight_fn(self._report_ts, self._user2ts[user_id]),
                          np.log(self._report_ts - self._user2ts[user_id])]
                for user_id in user_ids}
