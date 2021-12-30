import json
from os.path import join
from typing import List, Set, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.objects.frame import Frame
from src.data.objects.stack import Stack
from src.data.readers.annotation_reader import AnnotationLoader
from src.features.aggregators import Aggregator
from src.features.features_base import OverallFeature
from src.features.features_mapper import FeaturesMapper
from src.preprocess.entry_coders import Stack2FrameNames, Stack2FrameFileNames, Stack2FrameSubsystems, Entry2Seq
from src.preprocess.seq_coder import SeqCoder
from src.preprocess.tokenizers import FrameTokenizer


def _entry2seq(name_type: str) -> Entry2Seq:
    if name_type == "name":
        return Stack2FrameNames(rem_equals=False)
    elif name_type == "file_name":
        return Stack2FrameFileNames(rem_equals=False)
    elif name_type == "subsystem":
        return Stack2FrameSubsystems(rem_equals=False)
    else:
        raise ValueError("Wrong name type!")


def _non_empty_indices(frames: List[Frame], name_type: str) -> List[int]:
    indices = []
    for i, frame in enumerate(frames):
        if name_type == "name":
            name = frame.raw_frame.name
        elif name_type == "file_name":
            name = frame.raw_frame.file_name
        elif name_type == "subsystem":
            name = frame.raw_frame.subsystem
        else:
            raise ValueError("Wrong name type!")

        if name is not None:
            indices.append(i)

    return indices


def _identity(x):
    return x


class FrameIdf:
    def __init__(self, name_type: str):
        self._fitted = False
        self._name_type = name_type
        self._tokenizer = SeqCoder(_entry2seq(name_type), FrameTokenizer())
        self._tfidf = TfidfVectorizer(tokenizer=_identity, lowercase=False)
        self._default_idf = 0.

    def _tokenize(self, stack: Stack) -> List[str]:
        return [token.value for token in self._tokenizer.to_seq(stack)]

    def fit(self, stacks: List[Stack]) -> "FrameIdf":
        if self._fitted:
            print(f"FrameIdf is already fitted, fit skipped!")
            return self

        texts = [self._tokenize(stack) for stack in stacks]
        self._tfidf.fit(texts)
        self._default_idf = 1 + np.log(len(stacks) / 1)
        self._fitted = True
        return self

    def frames_idf(self, stack: Stack) -> List[float]:
        indices = _non_empty_indices(stack.frames, self._name_type)
        seq = self._tokenize(stack)
        assert len(seq) == len(indices)

        idf = [0.] * len(stack)
        for i, token in zip(indices, seq):
            if token in self._tfidf.vocabulary_:
                idf[i] = self._tfidf.idf_[self._tfidf.vocabulary_[token]]
            else:
                idf[i] = self._default_idf

        return idf


class OverallFeaturesBuilder(Aggregator):
    def __init__(self, annotation_loader: AnnotationLoader, features_list: List[OverallFeature]):
        super().__init__(features_list)
        self._fitted = False
        self._features_list = features_list
        self._annotation_loader = annotation_loader
        self._idf = {f"{key}_idf": FrameIdf(key) for key in ["name", "file_name", "subsystem"]}

    def fit(self, stacks: List[Stack]) -> "OverallFeaturesBuilder":
        if self._fitted:
            print("Stack features maker is already fitted, fit call skipped")
            return self

        for idf in self._idf.values():
            idf.fit(stacks)

        self._fitted = True
        return self

    def __call__(self, stack: Stack, user_ids: Set[int]) -> Dict[int, List[float]]:
        kwargs = {"annotation_loader": self._annotation_loader}
        kwargs.update(self._idf)
        features = {user_id: [] for user_id in user_ids}

        for feature_builder in self._features_list:
            built_features = feature_builder(stack, user_ids, **kwargs)
            for user_id in user_ids:
                features[user_id].extend(built_features[user_id])

        return features

    def set_loaders(self, annotation_loader: AnnotationLoader):
        self._annotation_loader = annotation_loader

    @staticmethod
    def from_dir(annotation_loader: AnnotationLoader, dir: str) -> "OverallFeaturesBuilder":
        return OverallFeaturesBuilder(annotation_loader,
                                      [FeaturesMapper.get_overall_feature(feature["name"], feature["params"])
                                       for feature in json.load(open(join(dir, "overall_features.json")))])
