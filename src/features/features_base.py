from abc import abstractmethod
from typing import List, Dict, Any, Optional, Set

from src.data.objects.frame import Frame
from src.data.objects.stack import Stack


class Feature:
    def __init__(self, dim: int, name: str, params: Optional[Dict[str, Any]] = None):
        self._dim = dim
        self._name = name
        self._params = params if params else {}

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim


class AnnotationHandler(Feature):
    """
    Computes annotation based feature line by line.
    """

    @abstractmethod
    def init(self, report_ts: int, frame: Frame):
        raise NotImplementedError

    @abstractmethod
    def handle_line(self, line_num: int, commit_hash: str, author: int, ts: int):
        raise NotImplementedError

    @abstractmethod
    def build(self, user_ids: Set[int]) -> Dict[int, List[float]]:
        raise NotImplementedError


class OverallFeature(Feature):
    @abstractmethod
    def __call__(self, stack: Stack, user_ids: Set[int], **kwargs) -> Dict[int, List[float]]:
        raise NotImplementedError
