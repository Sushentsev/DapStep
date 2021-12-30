from collections import Counter
from typing import List, Dict, Set

from src.data.objects.frame import Frame
from src.features.features_base import AnnotationHandler


class NormalizedNumberEditedLinesWindowHandler(AnnotationHandler):
    """
    Normalized (/ window_len) number of edited lines in window [error_line - up; error_line + down].
    """

    def __init__(self, up: int = 10, down: int = 10):
        super().__init__(2, "norm_num_edited_lines_window", {"up": up, "down": down})
        self._up = up
        self._down = down
        self._error_line = None
        self._num_edited_lines = None

    def init(self, report_ts: int, frame: Frame):
        self._error_line = frame.raw_frame.line_num - 1
        self._num_edited_lines = Counter()

    def handle_line(self, line_num: int, commit_hash: str, author: int, ts: int):
        if self._error_line - self._up <= line_num <= self._error_line + self._down:
            self._num_edited_lines[author] += 1

    def build(self, user_ids: Set[int]) -> Dict[int, List[float]]:
        max_edited_lines = max(self._num_edited_lines[user_id] for user_id in user_ids) + 1
        return {user_id: [self._num_edited_lines[user_id] / (self._down + self._up + 1),
                          self._num_edited_lines[user_id] / max_edited_lines]
                for user_id in user_ids}
