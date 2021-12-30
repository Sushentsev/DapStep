from copy import deepcopy
from typing import Dict, Set

from src.data.objects.frame import Frame
from src.data.readers.annotation_reader import AnnotationLoader
from src.stack_builders.base import FrameBuilder


class EntireFileFrameBuilder(FrameBuilder):
    def __init__(self, annotation_loader: AnnotationLoader):
        self._annotation_loader = annotation_loader

    def __call__(self, report_ts: int, frame: Frame, user_ids: Set[int]) -> Dict[int, Frame]:
        raw_frame = frame.raw_frame
        if raw_frame.file_name and raw_frame.commit_hash and raw_frame.line_num:
            annotation = self._annotation_loader(raw_frame.commit_hash, raw_frame.file_name)
            if annotation and raw_frame.line_num - 1 < len(annotation):
                annotation_authors = set(annotation.author[annotation.ts <= report_ts]) & user_ids
                return {author: deepcopy(frame) for author in annotation_authors}

        return {}
