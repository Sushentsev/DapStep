from abc import ABC
from typing import List, Dict, Set, Tuple, Any

from src.data.objects.annotation import Annotation
from src.data.objects.frame import Frame
from src.features.features_base import AnnotationHandler, Feature


class Aggregator(ABC):
    """
    Aggregates list of features in one feature vector.
    """

    def __init__(self, features: List[Feature]):
        self._features = [(feature.name, feature.params) for feature in features]
        self._dim = sum(feature.dim for feature in features)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def features_params(self) -> List[Tuple[str, Dict[str, Any]]]:
        return self._features


class AnnotationHandlersAggregator(Aggregator):
    """
    Aggregates several annotation handlers features into one feature vector.
    """

    def __init__(self, handlers: List[AnnotationHandler]):
        super().__init__(handlers)
        self._handlers = handlers

    def __call__(self, report_ts: int, frame: Frame, annotation: Annotation,
                 user_ids: Set[int]) -> Dict[int, List[float]]:
        for handler in self._handlers:
            handler.init(report_ts, frame)

        for line_num, (c_hash, author, line_ts) in enumerate(zip(annotation.commit_hash,
                                                                 annotation.author,
                                                                 annotation.ts)):
            if line_ts <= report_ts:
                for handler in self._handlers:
                    handler.handle_line(line_num, c_hash, author, line_ts)

        features = {user_id: [] for user_id in user_ids}
        for handler in self._handlers:
            built_features = handler.build(user_ids)
            for user_id in user_ids:
                features[user_id].extend(built_features[user_id])

        return features

