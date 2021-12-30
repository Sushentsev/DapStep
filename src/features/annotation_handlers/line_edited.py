from typing import List, Dict, Set

from src.data.objects.frame import Frame
from src.features.features_base import AnnotationHandler


class LineEditedHandler(AnnotationHandler):
    """
    Whether error line was modified by user.
    """

    def __init__(self):
        super().__init__(1, "line_edited")
        self._error_line = None
        self._author = None

    def init(self, report_ts: int, frame: Frame):
        self._error_line = frame.raw_frame.line_num - 1
        self._author = None

    def handle_line(self, line_num: int, commit_hash: str, author: int, ts: int):
        if line_num == self._error_line:
            self._author = author

    def build(self, user_ids: Set[int]) -> Dict[int, List[float]]:
        return {user_id: ([1] if user_id == self._author else [0]) for user_id in user_ids}
