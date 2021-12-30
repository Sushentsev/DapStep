import argparse
from os.path import join
from typing import Dict, Any

from sklearn.preprocessing import LabelEncoder

from src.models.encoders.tfidf import TfIdfComputer
from src.scripts.constructors import seq_coder_ctor, sklearn_model_ctor
from src.scripts.data_utils import load_data, save_pickle, load_config


def train(config: Dict[str, Any]):
    print(config)
    reports_dir, labels_path = join(config["data_dir"], "reports"), join(config["data_dir"], "labels.csv")
    train_stacks, _, _, y_train, _, _ = load_data(reports_dir, labels_path, **config["data_split"])

    vectorizer = TfIdfComputer(seq_coder_ctor(**config["coder"])).fit(train_stacks)
    label_encoder = LabelEncoder().fit(y_train)

    train_stacks = vectorizer.transform(train_stacks)
    y_train = label_encoder.transform(y_train)
    model = sklearn_model_ctor(config["model"]).fit(train_stacks, y_train)

    print(f"Train size: {len(train_stacks)}")

    if "save_dir" in config:
        save_dir = config["save_dir"]
        save_pickle(vectorizer, join(save_dir, "vectorizer.pkl"))
        save_pickle(label_encoder, join(save_dir, "label_encoder.pkl"))
        save_pickle(model, join(save_dir, "model.pkl"))
        print(f"Saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="../configs/sklearn_classification.yaml", type=str)
    args = parser.parse_args()
    train(load_config(args.config_path))
