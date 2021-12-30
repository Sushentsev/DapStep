import argparse
from os.path import join
from typing import Dict, Any, List, Tuple

import torch
from sklearn.preprocessing import LabelEncoder
from torch import optim

from src.model_selection.split import remove_unseen
from src.models.losses import CrossEntropyLoss
from src.preprocess.token import Token
from src.scripts.constructors import seq_coder_ctor, classification_model_ctor
from src.scripts.data_utils import load_config, load_data, save_pickle
from src.train.loaders.classification_dataset import sampler
from src.train.trainer import Trainer
from src.utils import device


def choose_non_empty(stacks: List[List[Token[int]]], y: List[int]) -> Tuple[List[List[Token[int]]], List[int]]:
    non_empty_indices = [i for i, stack in enumerate(stacks) if len(stack) > 0]
    return [stacks[i] for i in non_empty_indices], [y[i] for i in non_empty_indices]


def train(config: Dict[str, Any]):
    print(config)
    reports_dir, labels_path = join(config["data_dir"], "reports"), join(config["data_dir"], "labels.csv")
    train_stacks, val_stacks, _, y_train, y_val, _ = load_data(reports_dir, labels_path, **config["data_split"])

    seq_coder = seq_coder_ctor(**config["coder"]).fit(train_stacks)
    label_encoder = LabelEncoder().fit(y_train)

    print(f"Train size: {len(train_stacks)}")
    print(f"Val size: {len(val_stacks)}")
    print(f"Vocab size: {len(seq_coder)}")

    train_stacks, val_stacks = seq_coder.transforms(train_stacks), seq_coder.transforms(val_stacks)

    train_stacks, y_train = choose_non_empty(train_stacks, y_train)
    val_stacks, y_val = choose_non_empty(val_stacks, y_val)
    val_stacks, y_val = remove_unseen(set(y_train), val_stacks, y_val)

    y_train, y_val = label_encoder.transform(y_train), label_encoder.transform(y_val)

    norm = len(val_stacks) / config["data_split"]["val_size"] if config["data_split"]["val_size"] > 0 else 0
    print(f"Val norm: {round(norm, 3)}")
    print(f"Num classes: {len(set(y_train))}")

    train_sampler = sampler(train_stacks, y_train, config["batch_size"], shuffle=True)
    val_sampler = sampler(val_stacks, y_val, config["batch_size"], shuffle=False)

    model = classification_model_ctor(len(seq_coder), len(set(y_train)), config["model"]).to(device)
    optimizer = optim.Adam(model.parameters(), **config["optimizer"])
    trainer = Trainer(model, optimizer, CrossEntropyLoss(), **config["train"])
    trainer.run_train(train_sampler, val_sampler, norm)

    if "save_dir" in config:
        save_dir = config["save_dir"]
        torch.save(model, join(save_dir, "model.pt"))
        save_pickle(seq_coder, join(save_dir, "seq_coder.pkl"))
        save_pickle(label_encoder, join(save_dir, "label_encoder.pkl"))
        print(f"Saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="../configs/dl_classification.yaml", type=str)
    args = parser.parse_args()
    train(load_config(args.config_path))
