import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class DataSampler:
    def __init__(self, dataset: Dataset, shuffle: bool, batch_size: int):
        self._dataset = dataset
        self._shuffle = shuffle
        self._batch_size = batch_size

    def batches(self):
        indices = np.arange(len(self._dataset))

        if self._shuffle:
            indices = np.random.permutation(indices)

        for i in tqdm(range(0, len(self._dataset), self._batch_size)):
            if self._batch_size == 1:
                yield self._dataset[indices[i]]
            else:
                batches = [self._dataset[indices[j]] for j in range(i, min(i + self._batch_size, len(self._dataset)))]
                yield list(map(list, zip(*batches)))

    def __len__(self) -> int:
        return len(self._dataset)
