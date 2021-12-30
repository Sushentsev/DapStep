import json
from os.path import join
from typing import Dict, List

from src.data.objects.stack import Stack
from src.data.readers.annotation_reader import AnnotationLoader
from src.features.aggregators import AnnotationHandlersAggregator
from src.features.features_base import AnnotationHandler
from src.features.features_mapper import FeaturesMapper


class FrameFeaturesBuilder:
    def __init__(self, annotation_loader: AnnotationLoader, annotation_handlers: List[AnnotationHandler]):
        self._annotation_loader = annotation_loader
        self._annotation_aggregator = AnnotationHandlersAggregator(annotation_handlers)

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

                annotation_features = self._annotation_aggregator(stack.ts, frame, annotation, frame_user_ids)

                for user_id in frame_user_ids:
                    user_frame = user2stack[user_id].frames[frame_pointers[user_id]]
                    user_frame.data.features = annotation_features[user_id]
                    frame_pointers[user_id] += 1

    @property
    def dim(self) -> int:
        return self._annotation_aggregator.dim

    @staticmethod
    def from_dir(annotation_loader: AnnotationLoader, features_dir: str) -> "FrameFeaturesBuilder":
        annotation_handlers = [FeaturesMapper.get_annotation_handler(handler["name"], handler["params"])
                               for handler in json.load(open(join(features_dir, "annotation_handlers.json")))]

        return FrameFeaturesBuilder(annotation_loader, annotation_handlers)
