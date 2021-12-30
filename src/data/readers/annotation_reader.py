from functools import lru_cache
from os.path import join, exists
from typing import Optional

import pandas as pd

from src.data.objects.annotation import Annotation


class AnnotationLoader:
    def __init__(self, dir: str):
        self._dir = dir

    @lru_cache(maxsize=300_000)
    def __call__(self, commit_hash: Optional[str], file_name: Optional[str]) -> Optional[Annotation]:
        if commit_hash and file_name:
            annotation_path = join(self._dir, ":".join([commit_hash[:8], file_name, "annotation.csv"]))
            if exists(annotation_path):
                df = pd.read_csv(annotation_path)
                return Annotation(commit_hash=df.commit_hash.values, author=df.author.values, ts=df.timestamp.values)
