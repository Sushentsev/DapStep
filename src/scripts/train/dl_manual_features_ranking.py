import argparse
from os.path import join
from typing import Dict, Any

import torch
from torch import optim

from src.data.readers.annotation_reader import AnnotationLoader
from src.features.frame_features_builder import FrameFeaturesBuilder
from src.features.overall_features_builder import OverallFeaturesBuilder
from src.models.losses import RankNetLoss
from src.scripts.constructors import seq_coder_ctor, ranking_model_ctor
from src.scripts.data_utils import load_data, load_config, save_pickle
from src.scripts.stack_process import fixers_stacks_ctor
from src.stack_builders.builders import UserStackBuilder
from src.train.loaders.ranking_dataset import sampler
from src.train.trainer import Trainer
from src.utils import device


def train(config: Dict[str, Any]):
    print("Config:")
    print(config)
    reports_dir = join(config["data_dir"], "reports")
    files_dir = join(config["data_dir"], "files")
    labels_path = join(config["data_dir"], "labels.csv")

    annotation_loader = AnnotationLoader(files_dir)

    train_stacks, val_stacks, _, y_train, y_val, _ = load_data(reports_dir, labels_path, **config["data_split"])
    user_ids = set(y_train + y_val)

    seq_coder = seq_coder_ctor(**config["coder"]).fit(train_stacks)
    user_stack_builder = UserStackBuilder.get(annotation_loader)
    frame_features_builder = FrameFeaturesBuilder.from_dir(annotation_loader, config["features_dir"])
    overall_features_builder = OverallFeaturesBuilder.from_dir(annotation_loader, config["features_dir"]).fit(
        train_stacks)

    print(f"Vocabulary size: {len(seq_coder)}")
    print(f"Frame features dim: {frame_features_builder.dim}")
    print(f"Overall features dim: {overall_features_builder.dim}")

    train_stacks, train_fixers_stacks, train_overall_features, y_train = \
        fixers_stacks_ctor(train_stacks, y_train, user_ids,
                           user_stack_builder, frame_features_builder, seq_coder, overall_features_builder)
    val_stacks, val_fixers_stacks, val_overall_features, y_val = \
        fixers_stacks_ctor(val_stacks, y_val, user_ids,
                           user_stack_builder, frame_features_builder, seq_coder, overall_features_builder)

    val_norm = len(val_stacks) / config["data_split"]["val_size"] if config["data_split"]["val_size"] > 0 else 0

    print(f"Train size: {len(train_stacks)}")
    print(f"Val size: {len(val_stacks)}")
    print(f"Number of ranking assignees: {len(user_ids)}")
    print(f"Normalization factor: {round(val_norm, 3)}")

    train_sampler = sampler(train_stacks, train_fixers_stacks, train_overall_features, y_train, shuffle=True)
    val_sampler = sampler(val_stacks, val_fixers_stacks, val_overall_features, y_val, shuffle=False)

    model = ranking_model_ctor(len(seq_coder), frame_features_builder.dim, overall_features_builder.dim,
                               manual_features=True, **config["model"]).to(device)
    optimizer = optim.Adam(model.parameters(), **config["optimizer"])
    trainer = Trainer(model, optimizer, RankNetLoss(normed=False), **config["train"])
    trainer.run_train(train_sampler, val_sampler, val_norm)

    if "save_dir" in config:
        save_dir = config["save_dir"]
        torch.save(model, join(save_dir, "model.pt"))
        save_pickle(seq_coder, join(save_dir, "seq_coder.pkl"))
        save_pickle(overall_features_builder, join(save_dir, "overall_features_builder.pkl"))
        print(f"Saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="../configs/dl_ranking.yaml", type=str)
    args = parser.parse_args()
    train(load_config(args.config_path))
