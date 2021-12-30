from os.path import join
from typing import Dict, List

import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.data.objects.stack import Stack
from src.models.classification.neural_classifiers import NeuralClassifier
from src.preprocess.seq_coder import SeqCoder
from src.scripts.data_utils import load_pickle
from src.utils import device


def predict(test_stacks: List[Stack], model: NeuralClassifier, seq_coder: SeqCoder,
            label_encoder: LabelEncoder) -> List[Dict[int, float]]:
    y_pred = []
    for stack in tqdm(test_stacks):
        stack_coded = seq_coder.transform(stack)
        if len(stack_coded) == 0:
            y_pred.append({dev: 0. for dev in label_encoder.classes_})
        else:
            model_scores = model.predict([stack_coded])[0]
            y_pred.append({dev: score for dev, score in zip(label_encoder.classes_, model_scores)})

    return y_pred


def scores(test_stacks: List[Stack], model_dir: str) -> List[Dict[int, float]]:
    seq_coder = load_pickle(join(model_dir, "seq_coder.pkl"))
    label_encoder = load_pickle(join(model_dir, "label_encoder.pkl"))
    model = torch.load(join(model_dir, "model.pt"), map_location=device).eval()
    return predict(test_stacks, model, seq_coder, label_encoder)
