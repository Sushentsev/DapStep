from collections import defaultdict
from typing import List, Set, Dict

import numpy as np

from src.data.objects.stack import Stack
from src.features.features_base import OverallFeature


class FirstEditedFrame(OverallFeature):
    def __init__(self):
        super().__init__(4, "first_edited_frame")

    def __call__(self, stack: Stack, user_ids: Set[int], **kwargs) -> Dict[int, List[float]]:
        annotation_loader = kwargs["annotation_loader"]
        first_frame = defaultdict(lambda: len(stack))

        num_annotated_frames = 1
        for frame_num, frame in enumerate(stack.frames, start=1):
            annotation = annotation_loader(frame.raw_frame.commit_hash, frame.raw_frame.file_name)
            if annotation:
                num_annotated_frames += 1
                for author in np.unique(annotation.author):
                    first_frame[author] = min(first_frame[author], frame_num)

        min_edited_frame = min(first_frame[user_id] for user_id in user_ids) + 1
        return {user_id: [first_frame[user_id], first_frame[user_id] / (len(stack) + 1),
                          first_frame[user_id] / num_annotated_frames, first_frame[user_id] / min_edited_frame]
                for user_id in user_ids}
