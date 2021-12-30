import pickle
from typing import List, Tuple, Dict, Any

import yaml

from src.data.objects.stack import Stack
from src.data.readers.stack_reader import assignee_data, StackLoader
from src.model_selection.split import train_test_split


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as file:
        return yaml.safe_load(file)


def load_pickle(path: str):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(data, path: str):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_data(reports_dir: str, labels_path: str,
              val_size: int, test_size: int) -> Tuple[List[Stack], List[Stack], List[Stack], List[int], List[int], List[int]]:
    stack_loader = StackLoader(reports_dir)
    stack_ids, y = assignee_data(reports_dir, labels_path, stack_loader)
    train_val_stack_ids, test_stack_ids, y_train_val, y_test = train_test_split(stack_ids, y, test_size, include_unseen=True)
    train_stack_ids, val_stack_ids, y_train, y_val = train_test_split(train_val_stack_ids, y_train_val, val_size,
                                                                      include_unseen=True)

    train_stacks = [stack_loader(stack_id) for stack_id in train_stack_ids]
    val_stacks = [stack_loader(stack_id) for stack_id in val_stack_ids]
    test_stacks = [stack_loader(stack_id) for stack_id in test_stack_ids]

    return train_stacks, val_stacks, test_stacks, y_train, y_val, y_test
