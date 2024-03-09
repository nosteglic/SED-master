#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Koichi Miyazaki (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import math
import os

os.environ["WANDB_API_KEY"] = '6109ea69f151b0fa881f2c3a60db2ce11e9b8838'
os.environ["WANDB_MODE"] = 'offline'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
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
from models.CRNN import SEDModel
from trainer_MT import MeanTeacherTrainer, MeanTeacherTrainerOptions
from transforms import ApplyLog, Compose, get_transforms
from torch.utils.data._utils.collate import default_collate

def my_collate(batch):
    transposed = zip(*batch)
    transposed_split = [q for q in transposed]

    original_part = zip(*transposed_split[0])
    original_result = [default_collate(samples) for samples in original_part]

    # events_batch = [transposed_split[1]]
    events_batch = transposed_split[1]
    if len(transposed_split) == 3:
        bg_batch = [default_collate(samples) for samples in zip(*transposed_split[2])]
        return original_result, events_batch, bg_batch
    return original_result, events_batch

def collect_stats(datasets, save_path):
    """Compute dataset statistics
    Args:
        datasets:
        save_path:
    Return:
        mean: (np.ndarray)
        std: (np.ndarray)
    """
    logging.info("compute dataset statistics")
    stats = {}
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=1)

        for x, _, _ in tqdm(dataloader):
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
    parser.add_argument("--config", default="config/dcase21_MT_CRNN.yaml", type=str, help="Config file in yaml format")
    parser.add_argument("--debugmode", default=True, action="store_true", help="Debugmode")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    # load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(int(cfg["seed"]))

    model_name = cfg["wandb"]["name"]
    model_name = model_name + "-" + cfg["seed"]
    cfg["wandb"]["name"] = model_name
    cfg["wandb"]["id"] = model_name

    wandb.init(config=cfg, **cfg["wandb"])
    exp_path = Path(f"exp/{cfg['wandb']['name']}")
    if not Path("exp").exists():
        Path("exp").mkdir()
    # if debug is true, enable to overwrite experiment
    if exp_path.exists():
        logging.warning(f"{exp_path} is already exist.")
        if args.debugmode:
            logging.warning("Note that experiment will be overwrite.")
        else:
            logging.info("Experiment is interrupted. Make sure exp_path will be unique.")
            sys.exit(0)
    exp_path.mkdir(exist_ok=True)
    Path(exp_path / "model").mkdir(exist_ok=True)
    Path(exp_path / "predictions").mkdir(exist_ok=True)
    Path(exp_path / "log").mkdir(exist_ok=True)
    Path(exp_path / "score").mkdir(exist_ok=True)

    # save config
    shutil.copy(args.config, (exp_path / "config.yaml"))
    shutil.copy("src/methods/trainer_MT.py", (exp_path / "trainer.py"))
    shutil.copy("src/methods/train_MT_CRNN.py", (exp_path / "train.py"))
    shutil.copy("src/models/concat_data.py", (exp_path / "concat_data.py"))
    shutil.copy("src/models/CRNN.py", (exp_path / "CRNN.py"))



    # get config
    data_root = cfg["data_root"]
    feat_cfg = cfg["feature"]

    # get df from meta
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

    if Path(f"stats/stats_CRNN.npz").exists():
        shutil.copy("stats/stats_CRNN.npz", f"{exp_path}/stats.npz")

    # collect dataset stats
    if Path(f"{exp_path}/stats.npz").exists():
        stats = np.load(
            f"{exp_path}/stats.npz",
        )
    else:
        kwargs_dataset = {
            "encode_function": encode_function,
            "transforms": Compose([ApplyLog()]),
        }
        train_sync_dataset = SEDDataset_synth(train_sync_df, data_dir=(feat_dir / "train/synthetic"), use_events = False, **kwargs_dataset)
        train_weak_dataset = SEDDataset(train_weak_df, data_dir=(feat_dir / "train/weak"), **kwargs_dataset)
        train_unlabel_dataset = SEDDataset(train_unlabel_df, data_dir=(feat_dir / "train/unlabel_in_domain"), **kwargs_dataset)
        stats = collect_stats(
            [train_sync_dataset, train_weak_dataset, train_unlabel_dataset],
            f"{exp_path}/stats.npz",
        )


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

    use_events = cfg["use_events"]
    use_bg = cfg['use_bg']
    train_sync_dataset = SEDDataset_synth(
        train_sync_df,
        data_dir=(feat_dir / "train/synthetic"),
        encode_function=encode_function,
        pooling_time_ratio=cfg["pooling_time_ratio"],
        transforms=train_transforms,
        twice_data=True,
        use_events = use_events,
        use_bg = use_bg,
        classes = classes,
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

    batch_size = cfg["batch_size"]
    if cfg["ngpu"] > 1:
        batch_size *= cfg["ngpu"]

    loader_train_sync = DataLoader(
        train_sync_dataset,
        batch_size=cfg["batch_size_sync"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    loader_train_real_weak = DataLoader(
        train_weak_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    loader_train_real_unlabel = DataLoader(
        train_unlabel_dataset,
        batch_size=batch_size * 2,
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    loader_valid = DataLoader(
        valid_dataset,
        batch_size=batch_size,
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

    model = SEDModel(n_class=len(classes), attention=cfg["model"]["attention"], gen_count=cfg["gen_count"],
                     cnn_kwargs=cfg["model"]["cnn"], rnn_kwargs=cfg["model"]["rnn"])
    ema_model = SEDModel(n_class=len(classes), attention=cfg["model"]["attention"], gen_count=cfg["gen_count"],
                     cnn_kwargs=cfg["model"]["cnn"], rnn_kwargs=cfg["model"]["rnn"])

    # Show network architecture details
    logging.info(model)
    logging.info(model.parameters())
    logging.info(f"model parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    wandb.watch(model)

    trainer_options = MeanTeacherTrainerOptions(**cfg["trainer_options"])
    trainer_options._set_validation_options(
        valid_meta=valid_meta,
        valid_audio_dir=cfg["valid_audio_dir"],
        max_len_seconds=cfg["max_len_seconds"],
        sample_rate=cfg["feature"]["sample_rate"],
        hop_size=cfg["feature"]["mel_spec"]["hop_size"],
        pooling_time_ratio=cfg["pooling_time_ratio"],
    )

    # set optimizer and lr scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(trainable_params, **cfg["optimizer_params"])
    else:
        import torch_optimizer as optim

        optimizer = getattr(optim, cfg["optimizer"])(trainable_params, **cfg["optimizer_params"])

    scheduler = getattr(torch.optim.lr_scheduler, cfg["scheduler"])(optimizer, **cfg["scheduler_params"])

    trainer = MeanTeacherTrainer(
        model_name=model_name,
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
        use_events=use_events,
        use_bg=use_bg,
    )

    trainer.run()


if __name__ == "__main__":
    main(sys.argv[1:])