import argparse
import logging
import os
import yaml
import random
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
from torch.utils.data import DataLoader

from loc_vad import activity_detection
from src.post_processing import ScoreDataset
from matplotlib import pyplot as plt
# from mplfonts import use_font
# use_font('Noto Serif CJK SC')#指定中文字体

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

    parser.add_argument("--exp_name", default="MT_21-07", type=str, help="exp name used for the training")
    parser.add_argument("--exp_name2", default="MTM_21-07", type=str, help="exp name used for the training")
    parser.add_argument("--debugmode", default=True, action="store_true", help="Debugmode")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")

    return parser.parse_args(args)

def extend_arr(arr, scale):
    new_arr = np.zeros((arr.shape[0] * scale, arr.shape[1]))
    for i in range(len(arr)):
        new_arr[i * scale: (i + 1) * scale, :] = arr[i, :]
    return new_arr    

def smooth(data, th_high=0.85, th_low=0.35, n_smooth=3, n_salt=1):
    smoothed_outs = np.zeros((data.shape[0], data.shape[1]))
    for k in range(10):
        bgn_fin_pairs = activity_detection(
            x=data[:, k],
            thres=th_high,
            low_thres=th_low,
            n_smooth=n_smooth,
            n_salt=n_salt)
        for pair in bgn_fin_pairs:
            smoothed_outs[pair[0]:pair[1], k] = data[pair[0]:pair[1], k]
    return smoothed_outs

@torch.no_grad()
def test(exp_path, exp_path2):
    h5_path = "{}/test/posterior.h5".format(exp_path)
    h5_path2 = "{}/test/posterior.h5".format(exp_path2)

    dataset = ScoreDataset(h5_path , has_label=True)
    dataset2 = ScoreDataset(h5_path2 , has_label=True)

    data_loader = DataLoader(dataset)

    loc_figs_path = 'figs/papers/'
    labels = ['Speech', 'Dog' , 'Cat', 'Alarm_bell_ringing', 'Dishes', 'Frying', 'Blender', 'Running_water', 'Vacuum_cleaner', 'Electric_shaver_toothbrush']
    event_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    for batch_idx, data in enumerate(data_loader):
        data_id = data["data_id"][0]
        sed_pred = dataset.dataset[data_id]["pred_strong"]
        sed_pred2 = dataset2.dataset[data_id]["pred_strong"]
        target = dataset.dataset[data_id]["target"]
        fig, axs = plt.subplots(3, 1, figsize=(14, 6), dpi=200, sharex=True)
        rm = axs[0].imshow(extend_arr(target.T, 8), cmap='YlOrBr', aspect='auto')
        axs[0].set_yticks(np.arange(4, 10 * 8 + 4, 8))
        axs[0].set_ylabel('ground-truth')
        axs[0].set_yticklabels(labels, rotation=35)
        axs[1].set_xticks(np.arange(0, 60, 6))
        axs[1].set_xticklabels(np.arange(10.0))
        axs[1].imshow(extend_arr(smooth(sed_pred).T, 8), cmap='YlOrBr', aspect='auto')
        axs[1].set_yticks(np.arange(4, 10 * 8 + 4, 8))
        axs[1].set_yticklabels(labels, rotation=35)
        axs[1].set_ylabel('Baseline21')
        axs[2].imshow(extend_arr(smooth(sed_pred2).T, 8), cmap='YlOrBr', aspect='auto')
        axs[2].set_yticks(np.arange(4, 10 * 8 + 4, 8))
        axs[2].set_yticklabels(labels, rotation=35)
        axs[2].set_ylabel('MMT-CRNN')
        
        cb = fig.colorbar(rm, cmap='YlOrBr', ax=axs)
        cb.ax.tick_params(size=14)
        fig.savefig(loc_figs_path + data_id + '.png', bbox_inches='tight')
        plt.close(fig)
       
def test_single(exp_path, exp_path2, id):
    h5_path = "{}/test/posterior.h5".format(exp_path)
    h5_path2 = "{}/test/posterior.h5".format(exp_path2)
    
    dataset = ScoreDataset(h5_path, has_label=True)
    dataset2 = ScoreDataset(h5_path2, has_label=True)

    loc_figs_path = 'figs/papers/'
    labels = ['Speech', 'Dog', 'Cat', 'Alarm_bell_ringing', 'Dishes', 'Frying', 'Blender', 'Running_water',
              'Vacuum_cleaner', 'Electric_shaver_toothbrush']
    event_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    data_id = id
    sed_pred = dataset.dataset[data_id]["pred_strong"]
    sed_pred2 = dataset2.dataset[data_id]["pred_strong"]

    target = dataset.dataset[data_id]["target"]
    fig, axs = plt.subplots(3, 1, figsize=(14, 6), dpi=200, sharex=True)
    # axs[0].set_title(file_name, fontdict={'fontsize': 16, 'family': 'Times New Roman'})
    fig.suptitle('[File ID] '+data_id, fontsize=16);
    rm = axs[0].imshow(extend_arr(target.T, 8), cmap='YlOrBr', aspect='auto')
    axs[0].set_yticks(np.arange(4, 10 * 8 + 4, 8))
    # axs[0].set_ylabel('真实标签', fontdict={'fontsize': 16, 'family': 'Noto Serif CJK SC'})
    # axs[0].set_yticklabels(labels, rotation=35, fontdict={'fontsize': 11, 'family': 'Times New Roman'})
    axs[0].set_ylabel('ground-truth', fontsize=12)
    axs[0].set_yticklabels(event_labels, rotation=35)

    
    axs[1].set_xticks(np.arange(0, 60, 6))
    axs[1].set_xticklabels(np.arange(10.0))
    axs[1].imshow(extend_arr(smooth(sed_pred).T, 8), cmap='YlOrBr', aspect='auto')
    axs[1].set_yticks(np.arange(4, 10 * 8 + 4, 8))
    axs[1].set_yticklabels(event_labels, rotation=35)
    axs[1].set_ylabel('Baseline21', fontsize=12)

    axs[2].imshow(extend_arr(smooth(sed_pred2).T, 8), cmap='YlOrBr', aspect='auto')
    axs[2].set_yticks(np.arange(4, 10 * 8 + 4, 8))
    axs[2].set_yticklabels(event_labels, rotation=35)
    axs[2].set_ylabel('MMT-CRNN', fontsize=12)
    axs[2].set_xlabel('time (s)')
    
    # cb = fig.colorbar(rm, cmap='YlOrBr', ax=axs)
    # cb.ax.tick_params(size=16)
    fig.savefig(loc_figs_path+data_id + '.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
        
def main(args):
    args = parse_args(args)

    exp_path = Path(f"{os.getcwd()}/exp/{args.exp_name}")
    assert exp_path.exists()

    exp_path2 = Path(f"{os.getcwd()}/exp/{args.exp_name2}")
    assert exp_path2.exists()

    # load config
    with open(exp_path / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]

    # test_single(exp_path, exp_path2, id='6CLqqCp8Jfo_138_148.wav') # hFhsNDQ1mEo_0_5 e8jhGkgHG44_0_10 6CLqqCp8Jfo_138_148
    test(exp_path, exp_path2)
    # test_single(exp_path, exp_name=args.exp_name, id='-0H7vrzA80U_4_14.wav')



# /YYqaGwN2epw_46_56.wav.png, zM515Ca0AiI_79_89.wav.png, 49AbSai8It4_635_645.wav.png, 3cqsXEzUdiY_248_258.wav.png
if __name__ == "__main__":
    main(sys.argv[1:])
