import argparse
from os.path import join
from typing import List, Dict

from src.evaluation.metrics import acc_top_k, mrr
from src.scripts.data_utils import load_data
from src.scripts.eval import naive_baseline_pred, dl_classification_pred, dl_ranking_pred, sklearn_classification_pred
from src.utils import set_seed


def eval_model(y_true_raw: List[int], y_pred_raw: List[Dict[int, float]]):
    y_true, y_pred = [], []

    for dev_id, scores in zip(y_true_raw, y_pred_raw):
        devs = list(scores.keys())
        y_pred.append([scores[dev] for dev in devs])
        y_true.append(devs.index(dev_id) if dev_id in devs else -1)

    k = [1, 5, 10]
    set_seed()
    print(f"Accuracy {k}: {[round(acc, 3) for acc in acc_top_k(y_true, y_pred, k)]}")
    print(f"MRR: {mrr(y_true, y_pred):.3f}")


def eval(data_dir: str, model_dir: str, features_dir: str, test_size: int):
    reports_dir, labels_path = join(data_dir, "reports"), join(data_dir, "labels.csv")
    _, _, test_stacks, y_train, y_val, y_test = load_data(reports_dir, labels_path, val_size=0, test_size=test_size)
    dev_pool = set(y_train + y_val + y_test)

    print(f"Test size: {len(test_stacks)}")
    print(f"Dev pool size: {len(dev_pool)}")

    y_pred = dl_ranking_pred.scores(test_stacks, dev_pool, data_dir, model_dir, features_dir, neural_features=False)
    eval_model(y_test, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--features_dir", type=str)
    parser.add_argument("--test_size", type=int)
    args = parser.parse_args()
    eval(args.data_dir, args.model_dir, args.features_dir, args.test_size)
