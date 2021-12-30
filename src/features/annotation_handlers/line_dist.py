from typing import List, Set, Dict

from src.data.objects.frame import Frame
from src.features.features_base import AnnotationHandler


class LineDistanceHandler(AnnotationHandler):
    """
    Distance to error line and normed by annotation length, max distance.
    """

    def __init__(self):
        super().__init__(3, "line_dist")
        self._error_line = None
        self._user2dist = None
        self._annotation_len = None

    def init(self, report_ts: int, frame: Frame):
        self._error_line = frame.raw_frame.line_num - 1
        self._user2dist = {}
        self._annotation_len = 1

    def handle_line(self, line_num: int, commit_hash: str, author: int, ts: int):
        self._annotation_len += 1
        curr_dist = abs(self._error_line - line_num)
        self._user2dist[author] = min(self._user2dist.get(author, curr_dist), curr_dist)

    def build(self, user_ids: Set[int]) -> Dict[int, List[float]]:
        distances = {user_id: self._user2dist.get(user_id, self._annotation_len) for user_id in user_ids}
        min_distance = min(distances.values()) + 1
        return {user_id: [distances[user_id],
                          distances[user_id] / self._annotation_len,
                          distances[user_id] / min_distance]
                for user_id in user_ids}
