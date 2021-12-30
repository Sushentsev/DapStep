from dataclasses import dataclass
from typing import TypeVar, Generic

from src.data.objects.frame import FrameData

T = TypeVar("T", str, int)


@dataclass
class Token(Generic[T]):
    value: T
    data: FrameData
