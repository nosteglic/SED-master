#!/usr/bin/env python
# encoding: utf-8

# Copyright 2020 Koichi Miyazaki (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import ast
import logging
import math
import os
import pickle
import random
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from baseline_utils.ManyHotEncoder import ManyHotEncoder
from dataset import SEDDataset
from models.Conformer import SEDModel as Conformer
from models.CRNN import SEDModel as CRNN
from post_processing import PostProcess
from trainer_MT import MeanTeacherTrainerOptions
from transforms import get_transforms


def seed_everything(seed):
    logging.info("random seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--debugmode", default=True, action="store_true", help="Debugmode")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("--on_test", default=True, help="Choose validation or eval/public datasets")
    parser.add_argument("--revalid", default=False, help="Revalid parameters")
    parser.add_argument("--mode", type=str, default="score", help="Model uses best score or best loss or best psds")

    return parser.parse_args(args)


def test(model, test_loader, output_dir, options, pp_params={}):
    post_process = PostProcess(model, test_loader, output_dir, options)
    post_process.show_best(pp_params)
    post_process.compute_psds()

def valid(model, valid_loader, exp_path, options):
    post_process = PostProcess(model, valid_loader, exp_path, options)
    pp_params = post_process.tune_all()
    with open(exp_path / "post_process_params.pickle", "wb") as f:
        pickle.dump(pp_params, f)
    post_process.compute_psds()


def main(args):
    args = parse_args(args)
    args.revalid = ast.literal_eval(args.revalid)
    args.on_test = ast.literal_eval(args.on_test)
    if args.revalid:
        args.on_test = False

    exp_path = Path(f"{os.getcwd()}/exp/{args.exp_name}")
    assert exp_path.exists()

    # load config
    with open(exp_path / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    feat_cfg = cfg["feature"]
    if args.revalid:
        test_meta = f"{data_root}/metadata/validation/validation.tsv"
        test_audio_dir = f"{data_root}/audio/validation/validation"
    else:
        test_meta = f"{data_root}/metadata/eval/public.tsv"
        test_audio_dir = f"{data_root}/audio/eval/public"

    test_df = pd.read_csv(test_meta, header=0, sep="\t")

    n_frams_per_sec = feat_cfg["sample_rate"] / feat_cfg["mel_spec"]["hop_size"]
    n_frames = math.ceil(cfg["max_len_seconds"] * n_frams_per_sec)

    # Note: assume that the same class used in the training is included at least once.
    classes = test_df.event_label.dropna().sort_values().unique()
    many_hot_encoder = ManyHotEncoder(labels=classes, n_frames=n_frames)
    encode_function = many_hot_encoder.encode_strong_df
    test_df.onset = test_df.onset * n_frams_per_sec
    test_df.offset = test_df.offset * n_frams_per_sec
    feat_dir = Path(
        f"{data_root}/features/sr{feat_cfg['sample_rate']}_n_mels{feat_cfg['mel_spec']['n_mels']}_"
        + f"n_fft{feat_cfg['mel_spec']['n_fft']}_hop_size{feat_cfg['mel_spec']['hop_size']}"
    )

    stats = np.load(
        f"{exp_path}/stats.npz",
    )

    norm_dict_params = {
        "mean": stats["mean"],
        "std": stats["std"],
        "mode": cfg["norm_mode"],
    }

    if cfg["ngpu"] > 1:
        cfg["batch_size"] *= cfg["ngpu"]

    test_transforms = get_transforms(
        cfg["data_aug"],
        nb_frames=n_frames,
        norm_dict_params=norm_dict_params,
        training=False,
        prob=0.0,
    )

    if args.revalid:
        data_dir = feat_dir / "validation/validation"
    else:
        data_dir = feat_dir / "eval/public"


    test_dataset = SEDDataset(
        test_df,
        data_dir=data_dir,
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=test_transforms,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    seed_everything(int(cfg["seed"]))

    if "model_type" in cfg:
        model_type = cfg["model_type"]
    else:
        model_type = cfg["wandb"]["name"].split("-")[0]

    if model_type == "Conformer":
        model = Conformer(n_class=len(classes), cnn_kwargs=cfg["model"]["cnn"], gen_count=cfg["gen_count"],
                          encoder_kwargs=cfg["model"]["encoder"])
    elif model_type == "CRNN":
        model = CRNN(n_class=len(classes), attention=cfg["model"]["attention"], gen_count=cfg["gen_count"],
                     cnn_kwargs=cfg["model"]["cnn"], rnn_kwargs=cfg["model"]["rnn"])

    if args.mode == "score":
        checkpoint = torch.load(exp_path / "model" / "model_best_score.pth")
    elif args.mode == "loss":
        checkpoint = torch.load(exp_path / "model" / "model_best_loss.pth")
    elif args.mode == "psds":
        checkpoint = torch.load(exp_path / "model" / "model_best_psds.pth")
    else:
        raise ValueError("score_or_loss - Choose score or loss")
    model.load_state_dict(checkpoint["state_dict"])

    trainer_options = MeanTeacherTrainerOptions(**cfg["trainer_options"])
    trainer_options._set_validation_options(
        valid_meta=test_meta,
        valid_audio_dir=test_audio_dir,
        max_len_seconds=cfg["max_len_seconds"],
        sample_rate=cfg["feature"]["sample_rate"],
        hop_size=cfg["feature"]["mel_spec"]["hop_size"],
        pooling_time_ratio=cfg["pooling_time_ratio"],
    )

    model = model.to(trainer_options.device)

    if args.on_test:
        output_dir = exp_path / "test"
    else:
        if args.revalid:
            valid(model, test_loader, exp_path, trainer_options)
            return
        else:
            output_dir = exp_path / "valid"

    with open(exp_path / "post_process_params.pickle", "rb") as f:
        pp_params = pickle.load(f)
    output_dir.mkdir(exist_ok=True)
    test(model, test_loader, output_dir, trainer_options, pp_params=pp_params)


if __name__ == "__main__":
    main(sys.argv[1:])
