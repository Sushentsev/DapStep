from typing import List, Tuple, Any

from torch.utils.data import Dataset

from src.preprocess.token import Token
from src.train.loaders.sampler import DataSampler

EncodedStack = List[Token[int]]


class RankingDataset(Dataset):
    def __init__(self, stacks: List[EncodedStack], fixers_stacks: List[List[EncodedStack]],
                 overall_features: List[List[List[float]]], y: List[int]):
        super().__init__()
        self._stacks = stacks
        self._fixers_stacks = fixers_stacks
        self._overall_features = overall_features
        self._y = y

    def __getitem__(self, item: int) -> Tuple[Any, ...]:
        return self._stacks[item], self._fixers_stacks[item], self._overall_features[item], self._y[item]

    def __len__(self) -> int:
        return len(self._stacks)


def sampler(stacks: List[EncodedStack], fixers_stacks: List[List[EncodedStack]],
            overall_features: List[List[List[float]]], y: List[int], shuffle: bool) -> DataSampler:
    dataset = RankingDataset(stacks, fixers_stacks, overall_features, y)
    return DataSampler(dataset, shuffle, 1)
