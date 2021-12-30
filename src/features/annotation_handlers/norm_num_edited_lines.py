from collections import Counter
from typing import List, Set, Dict

from src.data.objects.frame import Frame
from src.features.features_base import AnnotationHandler


class NormalizedNumberEditedLinesHandler(AnnotationHandler):
    """
    Normalized number of edited lines.
    """

    def __init__(self):
        super().__init__(2, "norm_num_edited_lines")
        self._annotation_len = None
        self._num_edited_lines = None

    def init(self, report_ts: int, frame: Frame):
        self._annotation_len = 1
        self._num_edited_lines = Counter()

    def handle_line(self, line_num: int, commit_hash: str, author: int, ts: int):
        self._annotation_len += 1
        self._num_edited_lines[author] += 1

    def build(self, user_ids: Set[int]) -> Dict[int, List[float]]:
        max_counts = max(self._num_edited_lines[user_id] for user_id in user_ids) + 1
        return {user_id: [self._num_edited_lines[user_id] / self._annotation_len,
                          self._num_edited_lines[user_id] / max_counts]
                for user_id in user_ids}
