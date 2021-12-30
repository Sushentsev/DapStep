from abc import abstractmethod, ABC
from typing import List

from src.data.objects.stack import Stack


class UnsupFeatures(ABC):
    @abstractmethod
    def fit(self, stacks: List[Stack]) -> "UnsupFeatures":
        raise NotImplementedError

    @abstractmethod
    def transform(self, stacks: List[Stack]) -> List[List[float]]:
        raise NotImplementedError

    def fit_transform(self, stacks: List[Stack]) -> List[List[float]]:
        return self.fit(stacks).transform(stacks)
