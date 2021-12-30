import random

import numpy as np
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
