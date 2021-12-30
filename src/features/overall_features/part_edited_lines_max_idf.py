from collections import Counter
from typing import List, Optional, Set, Dict

import numpy as np

from src.data.objects.stack import Stack
from src.features.features_base import OverallFeature


class PartEditedLinesMaxIdfFrame(OverallFeature):
    def __init__(self, top_k: Optional[int] = None):
        super().__init__(3, "part_edited_lines_max_idf_frame", {"top_k": top_k} if top_k else None)
        self._top_k = top_k
        self._names = ["name", "file_name", "subsystem"]

    def __call__(self, stack: Stack, user_ids: Set[int], **kwargs) -> Dict[int, List[float]]:
        annotation_loader = kwargs["annotation_loader"]
        part_lines_max_idf_frame = {user_id: [0] * len(self._names) for user_id in user_ids}

        if len(stack) == 0:
            return part_lines_max_idf_frame

        for name_index, name in enumerate(self._names):
            idf_computer = kwargs[f"{name}_idf"]
            frame = stack.frames[np.argmax(idf_computer.frames_idf(stack)[:self._top_k])]
            annotation = annotation_loader(frame.raw_frame.commit_hash, frame.raw_frame.file_name)
            if annotation:
                line_counts = Counter(annotation.author[annotation.ts <= stack.ts])

                for user_id in user_ids:
                    part_lines_max_idf_frame[user_id][name_index] = line_counts[user_id] / len(annotation)

        return part_lines_max_idf_frame
