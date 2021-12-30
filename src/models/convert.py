from typing import List

from src.preprocess.token import Token


def get_stack_values(stack: List[Token[int]]) -> List[int]:
    return [token.value for token in stack]


def get_stack_features(stack: List[Token[int]]) -> List[List[float]]:
    return [token.data.features for token in stack]
