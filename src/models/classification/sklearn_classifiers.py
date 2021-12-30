from typing import List

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from src.data.objects.stack import Stack
from src.models.encoders.base import UnsupFeatures


class SklearnModel:
    def __init__(self, features: UnsupFeatures, model):
        self._features = features
        self._model = model

    def fit(self, stacks: List[Stack], fixers: List[int]) -> "SklearnModel":
        self._features.fit(stacks)
        self._model.fit(self._features.transform(stacks), fixers)
        return self

    def predict(self, stacks: List[Stack]) -> List[int]:
        return self._model.predict(self._features.transform(stacks))

    def predict_proba(self, stacks: List[Stack]) -> List[List[float]]:
        return self._model.predict_proba(self._features.transform(stacks))


class RandomForestModel(SklearnModel):
    def __init__(self, features: UnsupFeatures):
        super().__init__(features, RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None))


class SGDClassifierModel(SklearnModel):
    def __init__(self, features: UnsupFeatures):
        super().__init__(features, SGDClassifier(C=0.))
