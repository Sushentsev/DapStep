from typing import Set, Dict

from src.data.objects.stack import Stack
from src.data.readers.annotation_reader import AnnotationLoader
from src.stack_builders.base import FrameBuilder
from src.stack_builders.frame_builders.file_based import EntireFileFrameBuilder


class UserStackBuilder:
    def __init__(self, frame_builder: FrameBuilder):
        self._frame_builder = frame_builder
        self._default_stack_id = -1

    def __call__(self, stack: Stack, user_ids: Set[int]) -> Dict[int, Stack]:
        user2stack = {user_id: Stack(self._default_stack_id, stack.ts) for user_id in user_ids}

        for frame in stack.frames:
            user2frame = self._frame_builder(stack.ts, frame, user_ids)
            for user_id, user_frame in user2frame.items():
                user2stack[user_id].frames.append(user_frame)

        return user2stack

    @staticmethod
    def get(annotation_loader: AnnotationLoader) -> "UserStackBuilder":
        return UserStackBuilder(EntireFileFrameBuilder(annotation_loader))
