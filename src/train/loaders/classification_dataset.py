from typing import List, Tuple

from torch.utils.data import Dataset

from src.preprocess.token import Token
from src.train.loaders.sampler import DataSampler

EncodedStack = List[Token[int]]


class ClassificationDataset(Dataset):
    def __init__(self, stacks: List[EncodedStack], y: List[int]):
        super().__init__()
        self._stacks = stacks
        self._y = y

    def __getitem__(self, item: int) -> Tuple[EncodedStack, int]:
        return self._stacks[item], self._y[item]

    def __len__(self) -> int:
        return len(self._stacks)


def sampler(stacks: List[EncodedStack], y: List[int], batch_size: int, shuffle: bool) -> DataSampler:
    dataset = ClassificationDataset(stacks, y)
    return DataSampler(dataset, shuffle, batch_size)
