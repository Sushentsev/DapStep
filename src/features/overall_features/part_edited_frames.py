from collections import Counter
from typing import List, Optional, Set, Dict

import numpy as np

from src.data.objects.stack import Stack
from src.features.features_base import OverallFeature


class PartEditedFrames(OverallFeature):
    def __init__(self, top_k: Optional[int] = None):
        super().__init__(3, "part_edited_frames", {"top_k": top_k} if top_k else {})
        self._top_k = top_k

    def __call__(self, stack: Stack, user_ids: Set[int], **kwargs) -> Dict[int, List[float]]:
        annotation_loader = kwargs["annotation_loader"]
        num_edited_frames = Counter()
        num_frames = len(stack.frames[:self._top_k]) + 1
        num_annotated_frames = 1
        for frame in stack.frames[:self._top_k]:
            annotation = annotation_loader(frame.raw_frame.commit_hash, frame.raw_frame.file_name)
            if annotation:
                num_annotated_frames += 1
                for author in np.unique(annotation.author):
                    num_edited_frames[author] += 1

        max_num_edited_frames = max(num_edited_frames[user_id] for user_id in user_ids) + 1
        return {user_id: [num_edited_frames[user_id] / num_frames,
                          num_edited_frames[user_id] / num_annotated_frames,
                          num_edited_frames[user_id] / max_num_edited_frames]
                for user_id in user_ids}
