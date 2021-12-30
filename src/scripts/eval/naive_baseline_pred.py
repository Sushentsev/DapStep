from os.path import join
from typing import Dict, List, Set

from src.data.objects.stack import Stack
from src.data.readers.annotation_reader import AnnotationLoader
from src.models.naive.line_modified_classifier import LineModifiedClassifier
from src.models.weight_fns import exp_weight_fn


def scores(test_stacks: List[Stack], dev_pool: Set[int], data_dir: str) -> List[Dict[int, float]]:
    files_dir = join(data_dir, "files")
    annotation_loader = AnnotationLoader(files_dir)
    model = LineModifiedClassifier(dev_pool, annotation_loader, exp_weight_fn, 20)
    return model.predict(test_stacks)
