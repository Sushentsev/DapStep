from os.path import join
from typing import Dict, List

from sklearn.preprocessing import LabelEncoder

from src.data.objects.stack import Stack
from src.models.encoders.base import UnsupFeatures
from src.scripts.data_utils import load_pickle


def predict(test_stacks: List[Stack], model, vectorizer: UnsupFeatures,
            label_encoder: LabelEncoder) -> List[Dict[int, float]]:
    y_pred = []
    transformed = vectorizer.transform(test_stacks)
    model_scores = model.predict_proba(transformed)

    for probas in model_scores:
        y_pred.append({dev: proba for dev, proba in zip(label_encoder.classes_, probas)})

    return y_pred


def scores(test_stacks: List[Stack], model_dir: str) -> List[Dict[int, float]]:
    vectorizer = load_pickle(join(model_dir, "vectorizer.pkl"))
    label_encoder = load_pickle(join(model_dir, "label_encoder.pkl"))
    model = load_pickle(join(model_dir, "model.pkl"))

    return predict(test_stacks, model, vectorizer, label_encoder)
