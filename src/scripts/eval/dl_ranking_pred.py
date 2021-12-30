from os.path import join
from typing import Dict, List, Optional, Set

import torch
from tqdm import tqdm

from src.data.objects.stack import Stack
from src.data.readers.annotation_reader import AnnotationLoader
from src.features.frame_features_builder import FrameFeaturesBuilder
from src.features.neural_annotations_builder import NeuralAnnotationsBuilder
from src.features.overall_features_builder import OverallFeaturesBuilder
from src.models.base import NeuralModel
from src.preprocess.seq_coder import SeqCoder
from src.scripts.data_utils import load_pickle
from src.stack_builders.builders import UserStackBuilder
from src.utils import device


def predict(test_stacks: List[Stack], dev_pool: Set[int], model: NeuralModel, seq_coder: SeqCoder,
            user_stack_builder: UserStackBuilder, overall_features_builder: OverallFeaturesBuilder,
            frame_features_builder: FrameFeaturesBuilder,
            neural_annotations_builder: Optional[NeuralAnnotationsBuilder] = None) -> List[Dict[int, float]]:
    y_pred = []
    for stack in tqdm(test_stacks):
        dev2stack = user_stack_builder(stack, dev_pool)
        frame_features_builder.build(stack, dev2stack)
        if neural_annotations_builder is not None:
            neural_annotations_builder.build(stack, dev2stack)
        stack_tokens = seq_coder.transform(stack)
        dev2stack_tokens = {dev: seq_coder.transform(dev2stack[dev]) for dev in dev_pool}
        dev2overall_features = overall_features_builder(stack, dev_pool)
        all_empty = all([len(dev2stack_tokens[dev]) == 0 for dev in dev_pool])

        if all_empty:
            y_pred.append({dev: 0. for dev in dev_pool})
        else:
            non_empty_devs = [dev for dev in dev_pool if len(dev2stack_tokens[dev]) > 0]
            dev_stacks_tokens = [dev2stack_tokens[dev] for dev in non_empty_devs]
            overall_features = [dev2overall_features[dev] for dev in non_empty_devs]
            scores = model.predict(stack_tokens, dev_stacks_tokens, overall_features)

            preds = {dev: min(scores) for dev in dev_pool}
            preds.update({dev: score for dev, score in zip(non_empty_devs, scores)})
            y_pred.append(preds)

    return y_pred


def scores(test_stacks: List[Stack], dev_pool: Set[int], data_dir: str, model_dir: str,
           features_dir: str, neural_features: bool) -> List[Dict[int, float]]:
    files_dir = join(data_dir, "files")
    annotation_loader = AnnotationLoader(files_dir)

    seq_coder = load_pickle(join(model_dir, "seq_coder.pkl"))
    overall_features_builder = load_pickle(join(model_dir, "overall_features_builder.pkl"))
    overall_features_builder.set_loaders(annotation_loader)
    model = torch.load(join(model_dir, "model.pt"), map_location=device).eval()
    user_stack_builder = UserStackBuilder.get(annotation_loader)
    frame_features_builder = FrameFeaturesBuilder.from_dir(annotation_loader, features_dir)

    neural_annotations_builder = None
    if neural_features:
        neural_annotations_builder = NeuralAnnotationsBuilder(annotation_loader)

    print(f"Frame features: {frame_features_builder.dim}")
    print(f"Stack features: {overall_features_builder.dim}")

    return predict(test_stacks, dev_pool, model, seq_coder, user_stack_builder,
                   overall_features_builder, frame_features_builder, neural_annotations_builder)
