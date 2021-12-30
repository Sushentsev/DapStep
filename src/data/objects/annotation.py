from dataclasses import dataclass

import numpy as np


@dataclass
class Annotation:
    commit_hash: np.ndarray
    author: np.ndarray
    ts: np.ndarray

    def __len__(self) -> int:
        return len(self.commit_hash)
