from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import math
import random

class SEDDataset_synth(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        encode_function,
        pooling_time_ratio: int = 1,
        transforms=None,
        twice_data=False,
        classes: list = None,
        use_clean=False,
        use_concat=False,
        use_bg=False,
    ):
        self.df = df
        self.data_dir = data_dir
        self.encode_function = encode_function
        self.ptr = pooling_time_ratio
        self.transforms = transforms
        self.filenames = df.filename.drop_duplicates().values
        self.twice_data = twice_data

        if type(classes) in [np.ndarray, np.array]:
            self.classes = list(classes)

        self.use_clean = use_clean
        self.use_concat = use_concat
        self.use_bg = use_bg

        temp_root, feat_dir = os.path.split(os.path.abspath(os.path.join(self.data_dir, "../..")))
        if self.use_clean:
            self.clean_dir = Path(os.path.join((temp_root+"_clean"), feat_dir))
        if self.use_concat:
            self.event_dir = Path(os.path.join((temp_root+"_raw"), feat_dir))
            if self.use_bg:
                self.bg_dir = Path(os.path.join(temp_root+"_raw_bg", feat_dir))

        self._check_exist()


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        data_id = self.filenames[index]
        data = self._get_sample(data_id)
        label = self._get_label(data_id)

        clean_label = None
        if self.use_clean:
            _, clean_label = self._get_clean(data_id)

        concat_data = None
        concat_label = None
        if self.use_concat:
            event_dict = self._get_events(data_id, max_len=data.shape[0])
            concat_data, concat_label = self.concat_data(
                event_dict,
                n_class=len(self.classes),
                T=data.shape[0],
                F=data.shape[1]
            )
            if self.use_bg:
                bg_data = self._get_bg(data_id)
                concat_data = concat_data + bg_data

        if self.transforms is not None:
            data, label, clean_label = self.transforms((data, label, clean_label))
            if clean_label is not None:
                clean_label = clean_label[self.ptr // 2 :: self.ptr, :]
            if self.use_concat:
                concat_data, concat_label, _ = self.transforms((concat_data, concat_label))
                concat_label = concat_label[self.ptr // 2:: self.ptr, :]

        label = label[self.ptr // 2 :: self.ptr, :]



        result_dict={}
        if not self.twice_data:
            result_dict['origin'] = (
                torch.from_numpy(data).float().unsqueeze(0),
                torch.from_numpy(label).float(),
            )
        else:
            result_dict['origin'] = (
                torch.from_numpy(data[0]).float().unsqueeze(0),
                torch.from_numpy(data[1]).float().unsqueeze(0),
                torch.from_numpy(label).float(),
            )
        if self.use_clean:
            result_dict['clean'] = (
                torch.from_numpy(clean_label).float(),
            )
        if self.use_concat:
            if not self.twice_data:
                result_dict['concat'] = (
                    torch.from_numpy(concat_data).float().unsqueeze(0),
                    torch.from_numpy(concat_label).float(),
                )
            else:
                result_dict['concat'] = (
                    torch.from_numpy(concat_data[0]).float().unsqueeze(0),
                    torch.from_numpy(concat_data[1]).float().unsqueeze(0),
                    torch.from_numpy(concat_label).float(),
                )
        return result_dict

    def _check_exist(self):
        del_ids = []
        for i in range(len(self.filenames)):
            f = self.filenames[i]
            if not os.path.exists(self.data_dir / f.replace("wav", "npy")):
                del_ids.append(i)
        self.filenames = np.delete(self.filenames, del_ids)

    def _get_sample(self, filename):
        data = np.load(self.data_dir / filename.replace("wav", "npy")).astype(np.float32)
        return data


    def _get_label(self, filename):
        if {"onset", "offset", "event_label"}.issubset(self.df.columns):
            cols = ["onset", "offset", "event_label"]
            label = self.df[self.df.filename == filename][cols]
            if label.empty:
                label = []
        else:
            label = "empty"
            if "filename" not in self.df.columns:
                raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(self.df.columns)
                )
        if self.encode_function is not None:
            label = self.encode_function(label)
        return label

    def _get_clean(self, filename, cluster=50):
        filename = filename.replace("wav", "npy")
        clean_data = np.load(self.clean_dir / filename).astype(np.float32)
        clean_label = np.load(Path(str(self.clean_dir).replace("features", "labels")) / filename)
        clean_label = np.eye(cluster)[clean_label].astype(np.float32)
        return clean_data, clean_label

    def _get_bg(self, filename):
        fileid = filename[:-4]
        bg_data = None
        bg_root = f"{self.bg_dir}/{fileid}"
        if bg_root is not None and os.path.exists(bg_root):
            for bg_path in os.listdir(bg_root):
                if bg_path[-4:] == ".npy":
                    bg_data = np.load(f"{bg_root}/{bg_path}").astype(np.float32)
        return bg_data

    def _get_events(self, filename, max_len):
        fileid = filename[:-4]
        events_root = f"{self.event_dir}/{fileid}"
        events_list = []
        if events_root is not None and os.path.exists(events_root):
            for event_path in os.listdir(events_root):
                if event_path[-4:] == ".npy":
                    event_data = np.load(f"{events_root}/{event_path}").astype(np.float32)
                    if event_data.shape[0] > max_len // 2:
                        event_data = np.array_split(
                            event_data,
                            math.ceil(event_data.shape[0] * 2 / max_len)
                        )
                    event_name = "_".join(event_path.split("_")[1:])[:-4]
                    event_id = self.classes.index(event_name)
                    events_list.append(
                        {
                            "id": event_id,
                            "data": event_data
                        }
                    )
        return events_list

    def concat_data(self, events_dict, n_class, T=625, F=128):
        events_label = np.zeros((T, n_class))
        events_data = np.zeros((T, F))
        onset = 0
        split_len = T
        while split_len > 0:
            choice_data = None
            data = random.choice(events_dict)
            if type(data['data']) in [np.ndarray, np.array]:
                choice_data = data['data'][:split_len, :]
            elif type(data['data']) is list:
                choice_data = random.choice(data['data'])[:split_len, :]
            offset = choice_data.shape[0] + onset - 1
            events_label[onset:offset + 1, data['id']] = 1
            events_data[onset:offset + 1, :] = choice_data
            onset = offset + 1
            split_len = T - onset
        return events_data, events_label


class SEDDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        encode_function,
        pooling_time_ratio: int = 1,
        transforms=None,
        twice_data=False,
        n_frames_per_sec=0,
        is_valid=False,
    ):
        self.df = df
        self.data_dir = data_dir
        self.encode_function = encode_function
        self.ptr = pooling_time_ratio
        self.transforms = transforms
        self.filenames = df.filename.drop_duplicates().values
        self.features = {}
        self.twice_data = twice_data
        self.n_frames_per_sec = n_frames_per_sec
        self.offset = {}
        self._check_exist(is_valid)


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        data_id = self.filenames[index]
        data = self._get_sample(data_id)
        label = self._get_label(data_id)
        if self.transforms is not None:
            data, label, _ = self.transforms((data, label))

        label = label[self.ptr // 2 :: self.ptr, :]

        if not self.twice_data:
            return (
                torch.from_numpy(data).float().unsqueeze(0),
                torch.from_numpy(label).float(),
                data_id,
            )
        else:
            return (
                torch.from_numpy(data[0]).float().unsqueeze(0),
                torch.from_numpy(data[1]).float().unsqueeze(0),
                torch.from_numpy(label).float(),
                data_id,
            )

    def _check_exist(self, valid=False):
        del_ids = []
        for i in range(len(self.filenames)):
            f = self.filenames[i]
            npy_path = os.path.join(self.data_dir / f.replace("wav", "npy"))
            if not os.path.exists(npy_path):
                temp_strs = self.filenames[i].split('_')
                temp_offset = float(temp_strs[-1][:-4]) - 1
                temp_strs[-1] = str(temp_offset) + '00.wav'
                if not os.path.exists(self.data_dir / '_'.join(temp_strs).replace("wav", "npy")):
                    del_ids.append(i)
                else:
                    self.df.loc[self.df['filename'] == f, 'filename'] = '_'.join(temp_strs)
                    self.filenames[i] = '_'.join(temp_strs)
                    if valid:
                        self.offset[self.filenames[i]] = temp_offset * self.n_frames_per_sec

        self.filenames = np.delete(self.filenames, del_ids)


    def _get_sample(self, filename):
        if self.features.get(filename) is None:
            data = np.load((self.data_dir / filename.replace("wav", "npy"))).astype(np.float32)
            self.features[filename] = data
        else:
            data = self.features[filename]
        return data

    def _get_label(self, filename):
        if "event_labels" in self.df.columns or {"onset", "offset", "event_label"}.issubset(self.df.columns):
            if "event_labels" in self.df.columns:
                label = self.df[self.df.filename == filename]["event_labels"].values[0]
                if pd.isna(label):
                    label = []
                if type(label) is str:
                    if label == "":
                        label = []
                    else:
                        label = label.split(",")
            else:
                cols = ["onset", "offset", "event_label"]
                label = self.df[self.df.filename == filename][cols]
                if filename in self.offset and label.offset.iloc[-1] > self.offset[filename]:
                    label.offset.iloc[-1] = self.offset[filename]
                if label.empty:
                    label = []
        else:
            label = "empty"
            if "filename" not in self.df.columns:
                raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(self.df.columns)
                )
        if self.encode_function is not None:
            label = self.encode_function(label)
        return label