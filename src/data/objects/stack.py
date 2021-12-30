from dataclasses import dataclass, field
from typing import List

from src.data.objects.frame import Frame


@dataclass
class Stack:
    id: int
    ts: int
    frames: List[Frame] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)
