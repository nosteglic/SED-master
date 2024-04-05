#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Koichi Miyazaki (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import datetime
import logging
import math
import os

import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import shutil

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_utils.ManyHotEncoder import ManyHotEncoder
from dataset import SEDDataset, SEDDataset_synth
from models.Conformer import SEDModel as Conformer
from models.CRNN import SEDModel as CRNN
from trainer_MT import MeanTeacherTrainer, MeanTeacherTrainerOptions
from transforms import ApplyLog, Compose, get_transforms


def collect_stats(datasets, save_path):
    logging.info("compute dataset statistics")
    stats = {}
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=1)

        for x in tqdm(dataloader):
            if len(x) == 1:
                x = x['origin'][0]
            else:
                x = x[0]
            if len(stats) == 0:
                stats["mean"] = np.zeros(x.size(-1))
                stats["std"] = np.zeros(x.size(-1))
            stats["mean"] += x.numpy()[0, 0, :, :].mean(axis=0)
            stats["std"] += x.numpy()[0, 0, :, :].std(axis=0)
        stats["mean"] /= len(dataset)
        stats["std"] /= len(dataset)

    np.savez(save_path, **stats)

    return stats

def save_args(args, dest_dir, file_name="config.yaml"):
    import yaml

    print(yaml.dump(vars(args)))
    with open(os.path.join(dest_dir, file_name), "w") as f:
        f.write(yaml.dump(vars(args)))

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
    parser.add_argument("--model_type", type=str, default="test")

    return parser.parse_args(args)

def get_cfg(model_type):
    cfg_list = [
        "aug.yaml",
        "meta.yaml",
        f"{model_type}.yaml",
        "idea.yaml",
    ]

    cfg = {}
    for cfg_file in cfg_list:
        with open(os.path.join("config", cfg_file), "r") as f:
            cfg.update(yaml.safe_load(f))
    cfg['model_type'] = model_type
    cfg['exp_name'] = model_type
    if cfg['data_aug']['time_shift']['apply']:
        cfg['exp_name'] += "_timeshift"
    if cfg['mixup']:
        cfg['exp_name'] += '_mixup'
    if cfg['use_clean']:
        cfg['exp_name'] += "-clean"
        if cfg['alpha'] is not None:
            cfg['exp_name'] += f"-alpha{float(cfg['alpha'])}"
    if cfg['use_concat']:
        cfg['exp_name'] += "-concat"
        if cfg['use_mixup']:
            cfg['exp_name'] += "_mixup"
        if cfg['use_bg']:
            cfg['exp_name'] += "_bg"
        if cfg["use_contrast"]:
            cfg['exp_name'] += "-contrast"
            if cfg['beta'] is not None:
                cfg['exp_name'] += f"_beta{float(cfg['beta'])}"
    if cfg['other']:
        cfg['exp_name'] += cfg['other']
    cfg['exp_name'] += f"-seed{cfg['seed']}"

    if model_type != "test":
        print("*******Confirm your exp_name*******")
        print(cfg['exp_name'])
        print("***********************************")
        time.sleep(5)

    exp_path = Path(os.path.join(cfg["exp_root"], cfg["exp_name"]))
    if not Path(cfg["exp_root"]).exists():
        Path(cfg["exp_root"]).mkdir()
    exp_path.mkdir(exist_ok=True)
    Path(exp_path / "model").mkdir(exist_ok=True)
    Path(exp_path / "predictions").mkdir(exist_ok=True)

    with open(exp_path / "config.yaml", 'w') as file:
        yaml.dump(cfg, file, default_flow_style=False)
    return cfg, exp_path

def main(args):
    args = parse_args(args)
    model_type = args.model_type

    if model_type == "test":
        os.environ["WANDB_API_KEY"] = '6109ea69f151b0fa881f2c3a60db2ce11e9b8838'
        os.environ["WANDB_MODE"] = 'offline'
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    cfg, exp_path = get_cfg(model_type=model_type)

    seed_everything(int(cfg['seed']))

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    cfg['wandb']['name'] = f"{model_type}-{formatted_datetime}"
    cfg['wandb']['id'] = f"{model_type}-{formatted_datetime}"
    wandb.init(config=cfg, **cfg["wandb"])

    data_root = cfg["data_root"]
    feat_cfg = cfg["feature"]

    sync_meta = f"{data_root}/{cfg['sync_meta']}"
    weak_meta = f"{data_root}/{cfg['weak_meta']}"
    unlabel_meta = f"{data_root}/{cfg['unlabel_meta']}"
    valid_meta = f"{data_root}/{cfg['valid_meta']}"

    train_sync_df = pd.read_csv(sync_meta, header=0, sep="\t")
    train_weak_df = pd.read_csv(weak_meta, header=0, sep="\t")
    train_unlabel_df = pd.read_csv(unlabel_meta, header=0, sep="\t")
    valid_df = pd.read_csv(valid_meta, header=0, sep="\t")

    classes = valid_df.event_label.dropna().sort_values().unique()

    n_frames_per_sec = feat_cfg["sample_rate"] / feat_cfg["mel_spec"]["hop_size"]
    n_frames = math.ceil(cfg["max_len_seconds"] * n_frames_per_sec)

    many_hot_encoder = ManyHotEncoder(labels=classes, n_frames=n_frames)
    encode_function = many_hot_encoder.encode_strong_df

    train_sync_df.onset = train_sync_df.onset * n_frames_per_sec
    train_sync_df.offset = train_sync_df.offset * n_frames_per_sec

    valid_df.onset = valid_df.onset * n_frames_per_sec
    valid_df.offset = valid_df.offset * n_frames_per_sec

    feat_root = f"sr{feat_cfg['sample_rate']}_n_mels{feat_cfg['mel_spec']['n_mels']}_" \
                f"n_fft{feat_cfg['mel_spec']['n_fft']}_hop_size{feat_cfg['mel_spec']['hop_size']}"

    feat_dir = Path(f"{data_root}/features/{feat_root}")

    if Path(f"stats/stats_{model_type}.npz").exists():
        shutil.copy(f"stats/stats_{model_type}.npz", f"{exp_path}/stats.npz")

    if Path(f"{exp_path}/stats.npz").exists():
        stats = np.load(
            f"{exp_path}/stats.npz",
        )
    else:
        kwargs_dataset = {
            "encode_function": encode_function,
            "transforms": Compose([ApplyLog()]),
        }
        train_sync_dataset = SEDDataset_synth(train_sync_df, data_dir=(feat_dir / "train/synthetic"), **kwargs_dataset)
        train_weak_dataset = SEDDataset(train_weak_df, data_dir=(feat_dir / "train/weak"), **kwargs_dataset)
        train_unlabel_dataset = SEDDataset(train_unlabel_df, data_dir=(feat_dir / "train/unlabel_in_domain"), **kwargs_dataset)

        stats = collect_stats(
            [train_sync_dataset, train_weak_dataset, train_unlabel_dataset],
            f"{exp_path}/stats.npz",
        )
        shutil.copy(f"{exp_path}/stats.npz", f"stats/stats_{cfg['model_type']}.npz")

    norm_dict_params = {
        "mean": stats["mean"],
        "std": stats["std"],
        "mode": cfg["norm_mode"],
    }

    train_transforms = get_transforms(
        cfg["data_aug"],
        nb_frames=n_frames,
        norm_dict_params=norm_dict_params,
        training=True,
        prob=cfg["apply_prob"],
    )
    test_transforms = get_transforms(
        cfg["data_aug"],
        nb_frames=n_frames,
        norm_dict_params=norm_dict_params,
        training=False,
        prob=0.0,
    )

    train_sync_dataset = SEDDataset_synth(
        train_sync_df,
        data_dir=(feat_dir / "train/synthetic"),
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=train_transforms,
        twice_data=True,
        classes=classes,
        use_clean=cfg['use_clean'],
        use_concat=cfg['use_concat'],
        use_bg=cfg['use_bg'],
    )

    train_weak_dataset = SEDDataset(
        train_weak_df,
        data_dir=(feat_dir / "train/weak"),
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=train_transforms,
        twice_data=True,
    )
    train_unlabel_dataset = SEDDataset(
        train_unlabel_df,
        data_dir=(feat_dir / "train/unlabel_in_domain"),
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=train_transforms,
        twice_data=True,
    )

    valid_dataset = SEDDataset(
        valid_df,
        data_dir=(feat_dir / "validation/validation"),
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=test_transforms,
        n_frames_per_sec=n_frames_per_sec,
        is_valid=True,
    )

    if cfg["ngpu"] > 1:
        cfg['batch_size'] *= cfg["ngpu"]

    loader_train_sync = DataLoader(
        train_sync_dataset,
        batch_size=cfg['batch_size_sync'],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    loader_train_real_weak = DataLoader(
        train_weak_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    loader_train_real_unlabel = DataLoader(
        train_unlabel_dataset,
        batch_size=cfg['batch_size'] * 2,
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    loader_valid = DataLoader(
        valid_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    model = None
    ema_model = None
    if model_type == "Conformer":
        model = Conformer(
            n_class=len(classes), cnn_kwargs=cfg["model"]["cnn"], encoder_kwargs=cfg["model"]["encoder"],
            use_clean=cfg['use_clean']
        )
        ema_model = Conformer(
            n_class=len(classes), cnn_kwargs=cfg["model"]["cnn"], encoder_kwargs=cfg["model"]["encoder"],
            use_clean=cfg['use_clean']
        )
    elif cfg['model_type'] == "CRNN" or model_type == "test":
        model = CRNN(
            n_class=len(classes), attention=cfg["model"]["attention"],
            cnn_kwargs=cfg["model"]["cnn"], rnn_kwargs=cfg["model"]["rnn"],
            use_clean=cfg['use_clean']
        )
        ema_model = CRNN(
            n_class=len(classes), attention=cfg["model"]["attention"],
            cnn_kwargs=cfg["model"]["cnn"], rnn_kwargs=cfg["model"]["rnn"],
            use_clean=cfg['use_clean']
        )

    logging.info(model)
    logging.info(model.parameters())
    logging.info(f"model parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    wandb.watch(model)

    trainer_options = MeanTeacherTrainerOptions(use_mixup=cfg['mixup'], **cfg["trainer_options"])
    trainer_options._set_validation_options(
        valid_meta=valid_meta,
        valid_audio_dir=cfg["valid_audio_dir"],
        max_len_seconds=cfg["max_len_seconds"],
        sample_rate=cfg["feature"]["sample_rate"],
        hop_size=cfg["feature"]["mel_spec"]["hop_size"],
        pooling_time_ratio=cfg["pooling_time_ratio"],
    )

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(trainable_params, **cfg["optimizer_params"])
    else:
        import torch_optimizer as optim

        optimizer = getattr(optim, cfg["optimizer"])(trainable_params, **cfg["optimizer_params"])

    scheduler = getattr(torch.optim.lr_scheduler, cfg["scheduler"])(optimizer, **cfg["scheduler_params"])

    alpha, beta = None, None
    if cfg['alpha'] is not None:
        alpha = float(cfg['alpha'])
    if cfg['beta'] is not None:
        beta = float(cfg['beta'])
    trainer = MeanTeacherTrainer(
        model=model,
        ema_model=ema_model,
        loader_train_sync=loader_train_sync,
        loader_train_real_weak=loader_train_real_weak,
        loader_train_real_unlabel=loader_train_real_unlabel,
        loader_valid=loader_valid,
        optimizer=optimizer,
        scheduler=scheduler,
        exp_path=exp_path,
        pretrained=cfg["pretrained"],
        resume=cfg["resume"],
        trainer_options=trainer_options,
        alpha=alpha,
        beta=beta,
    )

    trainer.run()

    print(f"{cfg['exp_name']} is Done.")


if __name__ == "__main__":
    main(sys.argv[1:])