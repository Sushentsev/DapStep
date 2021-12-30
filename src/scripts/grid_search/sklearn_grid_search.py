import argparse
from os.path import join
from typing import List

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from src.data.objects.stack import Stack
from src.model_selection.grid_search import ClassificationGridSearch
from src.models.encoders.base import UnsupFeatures
from src.models.encoders.tfidf import TfIdfComputer
from src.scripts.constructors import seq_coder_ctor
from src.scripts.data_utils import load_data


def log_reg_grid_search(feature_coder: UnsupFeatures, train_stacks: List[Stack], val_stacks: List[Stack],
                        y_train: List[int], y_val: List[int], save_dir: str):
    params = {
        "loss": ["log"],
        "alpha": [10 ** i for i in range(-10, 2)]
    }
    print(f"Start LogReg")
    estimator = ClassificationGridSearch(SGDClassifier, params, feature_coder)
    estimator.estimate_params(train_stacks, val_stacks, y_train, y_val)
    estimator.save_results(join(save_dir, "log_reg.csv"))
    print(f"LogReg saved!")


def random_forest_grid_search(feature_coder: UnsupFeatures, train_stacks: List[Stack], val_stacks: List[Stack],
                              y_train: List[int], y_val: List[int], save_dir: str):
    params = {
        "n_estimators": [1, 5, 10, 25, 50, 100],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 3, 5, 10, 20]
    }
    print(f"Start RF")
    estimator = ClassificationGridSearch(RandomForestClassifier, params, feature_coder)
    estimator.estimate_params(train_stacks, val_stacks, y_train, y_val)
    estimator.save_results(join(save_dir, "rf.csv"))
    print(f"RF saved!")


def grid_search(data_dir: str, save_dir: str):
    train_stacks, val_stacks, _, y_train, y_val, _ = load_data(join(data_dir, "reports"), join(data_dir, "labels.csv"),
                                                               val_size=450, test_size=450)

    print(f"Train size: {len(train_stacks)}")
    print(f"Val size: {len(val_stacks)}")

    seq_coder = seq_coder_ctor(entry_coder_type="file_name", cased=True, trim_len=0, rem_equals=False)
    feature_coder = TfIdfComputer(seq_coder)
    log_reg_grid_search(feature_coder, train_stacks, val_stacks, y_train, y_val, save_dir)
    random_forest_grid_search(feature_coder, train_stacks, val_stacks, y_train, y_val, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    grid_search(args.data_dir, args.save_dir)
