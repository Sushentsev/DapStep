from collections import defaultdict
from typing import List, Set, Dict

from src.data.objects.frame import Frame
from src.features.features_base import AnnotationHandler


class NumberDifferentTsHandler(AnnotationHandler):
    """
    Number of different timestamps.
    """

    def __init__(self):
        super().__init__(2, "num_different_ts")
        self._user2ts = None

    def init(self, report_ts: int, frame: Frame):
        self._user2ts = defaultdict(set)

    def handle_line(self, line_num: int, commit_hash: str, author: int, ts: int):
        self._user2ts[author].add(ts)

    def build(self, user_ids: Set[int]) -> Dict[int, List[float]]:
        max_different_ts = max(len(self._user2ts[user_id]) for user_id in user_ids) + 1
        return {user_id: [len(self._user2ts[user_id]), len(self._user2ts[user_id]) / max_different_ts]
                for user_id in user_ids}
