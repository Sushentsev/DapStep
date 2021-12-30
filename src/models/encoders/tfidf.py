from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.objects.stack import Stack
from src.models.encoders.base import UnsupFeatures
from src.preprocess.seq_coder import SeqCoder


def identity_tokenizer(text: str):
    return text


class TfIdfComputer(UnsupFeatures):
    def __init__(self, coder: SeqCoder):
        self._coder = coder
        self._vectorizer = TfidfVectorizer(lowercase=False, tokenizer=identity_tokenizer)

    def fit(self, stacks: List[Stack]) -> "TfIdfComputer":
        self._coder.fit(stacks)
        tokens = [[token.value for token in self._coder.to_seq(stack)]
                  for stack in stacks]
        self._vectorizer.fit(tokens)
        return self

    def transform(self, stacks: List[Stack]) -> List[List[float]]:
        tokens = [[token.value for token in self._coder.to_seq(stack)]
                  for stack in stacks]
        return self._vectorizer.transform(tokens).toarray().tolist()
