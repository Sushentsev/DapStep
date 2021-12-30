from collections import Counter, defaultdict
from typing import List, Set, Dict

from src.data.objects.frame import Frame
from src.features.features_base import AnnotationHandler
from src.models.weight_fns import exp_weight_fn


class WeightedNormalizedNumberLinesLastEditedHandler(AnnotationHandler):
    """
    Normalized number of last edited lines * timestamp weight.
    """

    def __init__(self):
        super().__init__(2, "weight_norm_num_lines_last_edited")
        self._weight_fn = exp_weight_fn
        self._user2ts = None
        self._user_ts2num_lines = None
        self._annotation_len = None
        self._report_ts = None

    def init(self, report_ts: int, frame: Frame):
        self._report_ts = report_ts
        self._user2ts = defaultdict(lambda: 0)
        self._user_ts2num_lines = Counter()
        self._annotation_len = 1

    def handle_line(self, line_num: int, commit_hash: str, author: int, ts: int):
        self._annotation_len += 1
        self._user2ts[author] = max(self._user2ts[author], ts)
        self._user_ts2num_lines[(author, ts)] += 1

    def build(self, user_ids: Set[int]) -> Dict[int, List[float]]:
        edited_lines = {user_id: self._user_ts2num_lines[(user_id, self._user2ts[user_id])]
                        for user_id in user_ids}
        max_edited_lines = max(edited_lines.values()) + 1
        return {user_id: [edited_lines[user_id] / self._annotation_len * self._weight_fn(self._report_ts, self._user2ts[user_id]),
                          edited_lines[user_id] / max_edited_lines * self._weight_fn(self._report_ts, self._user2ts[user_id])]
                for user_id in user_ids}
