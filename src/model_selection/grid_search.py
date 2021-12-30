from typing import Dict, List, Any, Tuple

import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder

from src.data.objects.stack import Stack
from src.evaluation.metrics import acc_top_k
from src.models.encoders.base import UnsupFeatures
from src.utils import set_seed


def choose_non_empty_stacks(stacks: List[List[float]], fixers: List[int]) -> \
        Tuple[List[List[float]], List[int]]:
    new_stacks, new_fixers = [], []

    for stack, fixer in zip(stacks, fixers):
        if len(stack) > 0:
            new_stacks.append(stack)
            new_fixers.append(fixer)

    return new_stacks, new_fixers


class ClassificationGridSearch:
    def __init__(self, model_ctor, params: Dict[str, List[Any]], encoder: UnsupFeatures):
        self._model_ctor = model_ctor
        self._param_grid = list(ParameterGrid(params))
        self._encoder = encoder
        self._label_encoder = LabelEncoder()
        self._scores = []

    def estimate_params(self, train_stacks: List[Stack], val_stacks: List[Stack],
                        y_train: List[int], y_val: List[int]) -> "ClassificationGridSearch":
        self._scores = []
        train_stacks_coded = self._encoder.fit(train_stacks).transform(train_stacks)
        y_train_coded = self._label_encoder.fit_transform(y_train)

        val_stacks_coded, y_val_coded = [], []
        for stack, y in zip(val_stacks, y_val):
            if y in self._label_encoder.classes_:
                val_stacks_coded.append(self._encoder.transform([stack])[0])
                y_val_coded.append(self._label_encoder.transform([y])[0])

        print(f"Train {len(train_stacks)} | Val {len(val_stacks)} | Filtered Val {len(val_stacks_coded)}")
        k = len(val_stacks_coded) / len(val_stacks)
        print(f"Normalized coeff: {round(k, 2)}")

        for params in self._param_grid:
            set_seed()
            model = self._model_ctor(**params)
            model.fit(train_stacks_coded, y_train_coded)
            train_scores = acc_top_k(y_train_coded, model.predict_proba(train_stacks_coded), [1, 2, 3, 5, 10])
            val_scores = acc_top_k(y_val_coded, model.predict_proba(val_stacks_coded), [1, 2, 3, 5, 10])
            val_scores = [k * score for score in val_scores]

            print(f"Params {params}")
            print(f"Train score {[round(score, 2) for score in train_scores]} | "
                  f"Val score {[round(score, 2) for score in val_scores]}")

            self._scores.append((params,
                                 [round(score, 2) for score in train_scores],
                                 [round(score, 2) for score in val_scores]))

        return self

    def save_results(self, file_path: str):
        df = pd.DataFrame(self._scores, columns=["Params", "Train", "Val"])
        df.to_csv(file_path, index=False)

