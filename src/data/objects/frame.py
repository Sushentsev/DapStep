from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(eq=True)
class RawFrame:
    """
    Contains data from stacktrace.
    """
    name: str
    file_name: Optional[str]
    line_num: Optional[int]
    commit_hash: Optional[str]
    subsystem: Optional[str]


@dataclass
class FrameData:
    """
    Additional data for frame.
    Need for saving information during tokenization and so on.
    """
    special: bool = False
    features: List[float] = field(default_factory=list)
    annotations: List[List[float]] = field(default_factory=list)


@dataclass
class Frame:
    raw_frame: RawFrame
    data: FrameData

    def __eq__(self, other: "Frame") -> bool:
        return self.raw_frame == other.raw_frame
