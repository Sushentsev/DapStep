from abc import ABC, abstractmethod
from typing import Dict, Set

from src.data.objects.frame import Frame


class FrameBuilder(ABC):
    @abstractmethod
    def __call__(self, report_ts: int, frame: Frame, user_ids: Set[int]) -> Dict[int, Frame]:
        raise NotImplementedError
