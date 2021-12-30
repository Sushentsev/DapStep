from collections import Counter
from typing import List, Optional, Set, Dict

from src.data.objects.stack import Stack
from src.features.features_base import OverallFeature


class PartEditedErrorLines(OverallFeature):
    def __init__(self, top_k: Optional[int] = None):
        super().__init__(2, "part_edited_error_lines", {"top_k": top_k} if top_k else {})
        self._top_k = top_k

    def __call__(self, stack: Stack, user_ids: Set[int], **kwargs) -> Dict[int, List[float]]:
        annotation_loader = kwargs["annotation_loader"]

        num_frames = len(stack.frames[:self._top_k]) + 1
        edited_error_lines = Counter()
        for frame in stack.frames[:self._top_k]:
            annotation = annotation_loader(frame.raw_frame.commit_hash, frame.raw_frame.file_name)
            if annotation and frame.raw_frame.line_num and frame.raw_frame.line_num - 1 < len(annotation):
                error_line = frame.raw_frame.line_num - 1
                if annotation.ts[error_line] <= stack.ts:
                    edited_error_lines[annotation.author[error_line]] += 1

        max_edited_error_lines = max(edited_error_lines[user_id] for user_id in user_ids) + 1
        return {user_id: [edited_error_lines[user_id] / num_frames, edited_error_lines[user_id] / max_edited_error_lines]
                for user_id in user_ids}
