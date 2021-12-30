from typing import Dict, List

import numpy as np

from src.data.objects.annotation import Annotation
from src.data.objects.stack import Stack
from src.data.readers.annotation_reader import AnnotationLoader


def _to_year(milliseconds: np.ndarray) -> np.ndarray:
    return milliseconds / 1_000 / 60 / 60 / 24 / 365


def _get_user_annotation(user_id: int, annotation: Annotation,
                         error_line: int, report_ts: int) -> List[List[float]]:
    indices = np.argwhere(annotation.ts <= report_ts).squeeze()  # Choose right timestamp indices
    user_indices = indices[np.argwhere(annotation.author[indices] == user_id).flatten()]
    time_diff = np.log(_to_year(report_ts - annotation.ts[user_indices]))
    line_dist = np.abs(error_line - user_indices)
    sort_indices = np.argsort(time_diff)
    return list(map(list, zip(line_dist[sort_indices], time_diff[sort_indices])))


def _normalize(user2annotation: Dict[int, List[List[float]]]):
    max_line_dist = float("-inf")
    max_time = float("-inf")

    for annotation in user2annotation.values():
        for line_dist, time in annotation:
            max_line_dist = max(max_line_dist, line_dist)
            # max_time = max(max_time, time)

    for annotation in user2annotation.values():
        for line in annotation:
            line[0] /= (max_line_dist + 1)
            # line[1] /= (max_time + 1)


class NeuralAnnotationsBuilder:
    def __init__(self, annotation_loader: AnnotationLoader):
        self._annotation_loader = annotation_loader

    def build(self, stack: Stack, user2stack: Dict[int, Stack]):
        user_ids = set(user2stack.keys())
        frame_pointers = dict.fromkeys(user_ids, 0)

        for frame in stack.frames:
            frame_user_ids = set()
            for user_id in user_ids:
                if frame_pointers[user_id] < len(user2stack[user_id]) and \
                        user2stack[user_id].frames[frame_pointers[user_id]] == frame:
                    frame_user_ids.add(user_id)

            if len(frame_user_ids) > 0:
                annotation = self._annotation_loader(frame.raw_frame.commit_hash, frame.raw_frame.file_name)

                if (frame.raw_frame.line_num is not None) and annotation:
                    error_line = frame.raw_frame.line_num - 1
                    user2annotation = {user_id: _get_user_annotation(user_id, annotation, error_line, stack.ts)
                                       for user_id in frame_user_ids}
                    _normalize(user2annotation)

                    for user_id in frame_user_ids:
                        user_frame = user2stack[user_id].frames[frame_pointers[user_id]]
                        user_frame.data.annotations = user2annotation[user_id]
                        frame_pointers[user_id] += 1
