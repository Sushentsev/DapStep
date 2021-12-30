from typing import List, Callable, Optional, Dict, Set

from tqdm import tqdm

from src.data.objects.frame import Frame
from src.data.objects.stack import Stack
from src.data.readers.annotation_reader import AnnotationLoader


class LineModifiedClassifier:
    def __init__(self, user_ids: Set[int], annotation_loader: AnnotationLoader,
                 weight_fn: Callable[[int, int], float], top_k_frames: Optional[int] = None):
        self._user_ids = user_ids
        self._annotation_loader = annotation_loader
        self._weight_fn = weight_fn
        self._top_k_frames = top_k_frames

    def _frame_scores(self, frame: Frame, stack_ts: int) -> Dict[int, float]:
        scores = dict.fromkeys(self._user_ids, 0)
        frame = frame.raw_frame

        annotation = self._annotation_loader(frame.commit_hash, frame.file_name)
        if annotation and frame.line_num and frame.line_num - 1 < len(annotation):
            line_author = annotation.author[frame.line_num - 1]
            line_ts = annotation.ts[frame.line_num - 1]

            if line_author in self._user_ids and line_ts <= stack_ts:
                scores[line_author] += self._weight_fn(stack_ts, line_ts)

        return scores

    def _stack_scores(self, stack: Stack) -> Dict[int, float]:
        user_scores = dict.fromkeys(self._user_ids, 0)
        frames = stack.frames[:self._top_k_frames]

        for frame in frames:
            frame_scores = self._frame_scores(frame, stack.ts)
            for user_id in self._user_ids:
                user_scores[user_id] += frame_scores[user_id]

        return user_scores

    def predict(self, stacks: List[Stack]) -> List[Dict[int, float]]:
        return [self._stack_scores(stack) for stack in tqdm(stacks)]
