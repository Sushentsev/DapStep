from collections import Counter
from typing import List, Optional, Set, Dict

from src.data.objects.stack import Stack
from src.features.features_base import OverallFeature


class PartEditedLines(OverallFeature):
    def __init__(self, top_k: Optional[int] = None):
        super().__init__(2, "part_edited_lines", {"top_k": top_k} if top_k else None)
        self._top_k = top_k

    def __call__(self, stack: Stack, user_ids: Set[int], **kwargs) -> Dict[int, List[float]]:
        annotation_loader = kwargs["annotation_loader"]

        total_lines = 1
        num_edited_lines = Counter()
        for frame in stack.frames[:self._top_k]:
            annotation = annotation_loader(frame.raw_frame.commit_hash, frame.raw_frame.file_name)
            if annotation:
                num_edited_lines += Counter(annotation.author)
                total_lines += len(annotation)

        max_num_edited_lines = max(num_edited_lines[user_id] for user_id in user_ids) + 1
        return {user_id: [num_edited_lines[user_id] / total_lines,
                          num_edited_lines[user_id] / max_num_edited_lines] for user_id in user_ids}
