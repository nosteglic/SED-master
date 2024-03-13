import argparse
import logging
import subprocess
from pathlib import Path

import librosa
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm
import os

import sox
import pandas as pd

sox.core.sox_path = '/home/syx/local/bin/sox'


def resample_wav(cfg, src, dest):
    # sox_option = f'--norm={cfg["gain"]} -r {cfg["sample_rate"]} -c 1 -b 16 -t wav'
    sox_option = f'--norm={cfg["gain"]} -r {cfg["sample_rate"]} -c 1 -t wav'
    filter_option = f'highpass {cfg["highpass"]}'
    subprocess.run(f"sox -v 0.6 {src} {sox_option} {dest} {filter_option}", shell=True)


def feature_extraction(cfg, src, dest):
    y, sr = librosa.load(src, sr=None)
    mel_spec = calculate_mel_spec(y, sr, **cfg["mel_spec"])
    np.save(dest, mel_spec)

    y_duration = librosa.get_duration(filename=src)
    _, filename = os.path.split(src)


    return {"filename": filename, "duration": y_duration}


def calculate_mel_spec(x, fs, n_mels, n_fft, hop_size, fmin=None, fmax=None):
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    # Compute spectrogram
    ham_win = np.hamming(n_fft)

    spec = librosa.stft(x, n_fft=n_fft, hop_length=hop_size, window=ham_win, center=True, pad_mode="reflect")

    mel_spec = librosa.feature.melspectrogram(
        S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
        sr=fs,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=False,
        norm=None,
    )

    # if self.save_log_feature:
    #     mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
    mel_spec = mel_spec.T
    mel_spec = mel_spec.astype(np.float32)
    return mel_spec


def resample_wav_synth(cfg, need_resample_wav, wav_root, wav_dir, nj=1):
    if need_resample_wav:
        wav_dir.mkdir(parents=True, exist_ok=True)
        for x in ["train/synthetic", "train/weak", "train/unlabel_in_domain", "validation/validation", "eval/public"]:
            src_dir = Path(wav_root) / x
            dest_dir = wav_dir / x
            dest_dir.mkdir(parents=True, exist_ok=True)
            Parallel(n_jobs=nj)(
                delayed(resample_wav)(cfg, filename, (dest_dir / filename.name))
                for filename in
                tqdm(src_dir.glob("*.wav"), ncols=100, desc=x + " of wavs extract", total=len(os.listdir(src_dir)))
            )
    else:
        logging.info(f"{wav_dir} is already exists, resampling is skipped.")


def resample_wav_raw(cfg, need_resample_wav, wav_root, wav_dir, nj=1):
    if need_resample_wav:
        wav_dir.mkdir(parents=True, exist_ok=True)
        for root, dirs, files in os.walk(wav_root):
            for x in dirs:
                src_dir = Path(wav_root) / x
                dest_dir = wav_dir / x
                dest_dir.mkdir(parents=True, exist_ok=True)
                Parallel(n_jobs=nj)(
                    delayed(resample_wav)(cfg, filename, (dest_dir / filename.name))
                    for filename in
                    tqdm(src_dir.glob("*.wav"), ncols=100, desc=x + " of wavs extract", total=len(os.listdir(src_dir)))
                )
    else:
        logging.info(f"{wav_dir} is already exists, resampling is skipped.")


def recompute_feature_synth(cfg, need_recompute_feature, wav_dir, feat_dir, nj=1):
    if need_recompute_feature:
        feat_dir.mkdir(parents=True, exist_ok=True)
        for x in ["train/synthetic", "train/weak", "train/unlabel_in_domain", "validation/validation", "eval/public"]:
            src_dir = wav_dir / x
            dest_dir = feat_dir / x
            dest_dir.mkdir(parents=True, exist_ok=True)

            Parallel(n_jobs=nj)(
                delayed(feature_extraction)(cfg, filename, (dest_dir / filename.stem))
                for filename in
                tqdm(src_dir.glob("*.wav"), ncols=100, desc=x + " of feats extract", total=len(os.listdir(src_dir)))
            )
    else:
        logging.info(f"{feat_dir} is already exists, feature extraction is skipped.")

def recompute_feature_raw(cfg, need_recompute_feature, wav_dir, feat_dir, nj=1):
    if need_recompute_feature:
        feat_dir.mkdir(parents=True, exist_ok=True)
        for root, dirs, files in os.walk(wav_dir):
            for x in dirs:
                src_dir = wav_dir / x
                dest_dir = feat_dir / x
                dest_dir.mkdir(parents=True, exist_ok=True)

                dict_duration = Parallel(n_jobs=nj)(
                    delayed(feature_extraction)(cfg, filename, (dest_dir / filename.stem))
                    for filename in
                    tqdm(src_dir.glob("*.wav"), ncols=100, desc=x + " of feats extract", total=len(os.listdir(src_dir)))
                )

                df_duration = pd.DataFrame(dict_duration)
                df_duration.to_csv(dest_dir / "duration.tsv", index=False, sep="\t")
    else:
        logging.info(f"{feat_dir} is already exists, feature extraction is skipped.")

def ext_synth(cfg, nj, nee_resample_wav, need_recompute_feature):
    wav_root = f"{cfg['data_root']}/audio"
    feat_root = f"{cfg['data_root']}/features"
    cfg = cfg["feature"]
    wav_dir = Path(f"{wav_root}/wav/sr{cfg['sample_rate']}")

    resample_wav_synth(cfg=cfg, need_resample_wav=nee_resample_wav, wav_root=wav_root, wav_dir=wav_dir, nj=nj)

    feat_dir = Path(
        f"{feat_root}/sr{cfg['sample_rate']}"
        + f"_n_mels{cfg['mel_spec']['n_mels']}_n_fft{cfg['mel_spec']['n_fft']}_hop_size{cfg['mel_spec']['hop_size']}"
    )
    recompute_feature_synth(cfg=cfg, need_recompute_feature=need_recompute_feature, wav_dir=wav_dir, feat_dir=feat_dir, nj=nj)


def ext_raw(cfg, nj, nee_resample_wav, need_recompute_feature):
    super_root = os.path.split(cfg['data_root'])[0]
    # wav_root = f"{super_root}/sources_raw_targets"
    # feat_root = f"{cfg['data_root']}/feature_raw"
    wav_root = f"{super_root}/sources_raw_backgrounds"
    feat_root = f"{cfg['data_root']}/feature_raw_bg"

    cfg = cfg["feature"]
    # wav_dir = Path(f"{super_root}/wav_raw/sr{cfg['sample_rate']}")
    wav_dir = Path(f"{super_root}/wav_raw_bg/sr{cfg['sample_rate']}")

    resample_wav_raw(cfg=cfg, need_resample_wav=nee_resample_wav, wav_root=wav_root, wav_dir=wav_dir, nj=nj)

    feat_dir = Path(
        f"{feat_root}/sr{cfg['sample_rate']}"
        + f"_n_mels{cfg['mel_spec']['n_mels']}_n_fft{cfg['mel_spec']['n_fft']}_hop_size{cfg['mel_spec']['hop_size']}"
    )
    recompute_feature_raw(cfg=cfg, need_recompute_feature=need_recompute_feature, wav_dir=wav_dir, feat_dir=feat_dir, nj=nj)

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='config/dcase21_MT_Conformer.yaml')
    parser.add_argument("--nj", type=int, default=1)
    parser.add_argument("--nee_resample_wav", type=bool, default=False)
    parser.add_argument("--need_recompute_feature", type=bool, default=True)
    parser.add_argument("--extract_type", default="raw")
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    nj = args.nj
    nee_resample_wav = args.nee_resample_wav
    need_recompute_feature = args.need_recompute_feature
    # ext_synth(cfg=cfg, nj=nj)
    ext_raw(
        cfg=cfg,
        nj=nj,
        nee_resample_wav=nee_resample_wav,
        need_recompute_feature=need_recompute_feature)