from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import math
import random




# def encode_duration_df(events_df, labels):
#     if type(labels) in [np.ndarray, np.array]:
#         labels = labels.tolist()
#     for i, row in events_df.iterrows():
#         y = np.zeros((row.duration, len(labels)))
#         ind = labels.index(row.event_label)
#         y[:, ind] = 1
#
#     return events_df


class SEDDataset_synth(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        encode_function,
        pooling_time_ratio: int = 1,
        transforms=None,
        twice_data=False,
        use_events=False,
        classes: list = None,
        use_bg = False,
        gen_count=1,
    ):
        self.df = df
        self.data_dir = data_dir
        self.encode_function = encode_function
        self.ptr = pooling_time_ratio
        self.transforms = transforms
        self.filenames = df.filename.drop_duplicates().values
        self.twice_data = twice_data

        self.use_events = use_events
        self.use_bg = use_bg
        self.gen_count = gen_count

        if use_events or use_bg:
            temp_root, feat_dir = os.path.split(os.path.abspath(os.path.join(self.data_dir, "../..")))

            if use_events:
                self.event_dir = os.path.join(temp_root+"_raw", feat_dir)
            if use_bg:
                self.bg_dir = os.path.join(temp_root+"_raw_bg",feat_dir)


        if type(classes) in [np.ndarray, np.array]:
            self.classes = list(classes)

        self._check_exist()


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        data_id = self.filenames[index]
        data = self._get_sample(data_id)
        label = self._get_label(data_id)  # label - (625, 10)

        if self.use_bg:
            bg_data = self._get_bg(data_id)
        else:
            bg_data = None
        if self.use_events:
            events_dict = self._get_events(data_id, max_len=data.shape[0])
            events_data, events_label, events_len = self.concat_data(
                events_dict,
                n_class=len(self.classes),
                bg=bg_data,
                T=data.shape[0],
                F=data.shape[1]
            )

        if self.transforms is not None:
            data, label = self.transforms((data, label))
            if self.use_events:
                events_data, events_label = self.transforms((events_data, events_label))
            # data - 0 - (625, 128)
            # data - 1 - (625, 128)
            # label - (625, 10)

        # label pooling here because data augmentation may handle label (e.g. time shifting)
        # select center frame as a pooled label
        label = label[self.ptr // 2 :: self.ptr, :] # label - (156, 10)
        events_label = events_label[self.ptr // 2 :: self.ptr, :]
        events_len = [(i - self.ptr // 2) // self.ptr + 1 for i in events_len]

        # Return twice data with different augmentation if use mean teacher training
        if not self.twice_data:
            if self.use_events:
                return (
                    torch.from_numpy(data).float().unsqueeze(0),
                    torch.from_numpy(label).float(),
                    torch.from_numpy(events_data).float().unsqueeze(0),
                    torch.from_numpy(events_label).float().unsqueeze(0),
                    events_len
                )
            else:
                return (
                    torch.from_numpy(data).float().unsqueeze(0),
                    torch.from_numpy(label).float(),
                    data_id,
                )
        else:
            if self.use_events:
                return (
                    torch.from_numpy(data[0]).float().unsqueeze(0),
                    torch.from_numpy(data[1]).float().unsqueeze(0),
                    torch.from_numpy(label).float(),
                    torch.from_numpy(events_data[0]).float().unsqueeze(0),
                    torch.from_numpy(events_data[1]).float().unsqueeze(0),
                    torch.from_numpy(events_label).float(),
                    events_len
                )
            else:
                return (
                    torch.from_numpy(data[0]).float().unsqueeze(0),
                    torch.from_numpy(data[1]).float().unsqueeze(0),
                    torch.from_numpy(label).float(),
                    torch.from_numpy(events_label).float(),
                    data_id,
                )

    def _check_exist(self):
        del_ids = []
        for i in range(len(self.filenames)):
            f = self.filenames[i]
            if not os.path.exists(self.data_dir / f.replace("wav", "npy")):
                del_ids.append(i)
        self.filenames = np.delete(self.filenames, del_ids)

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

    def concat_data(self, events_dict, n_class, bg=None, T=625, F=128):
        events_label = np.zeros((T, n_class))
        events_data = np.zeros((T, F))
        events_len = [0] * self.gen_count
        for count in range(self.gen_count):
            random.shuffle(events_dict)
            onset = 0
            offset = 0
            for i, data in enumerate(events_dict):
                split_len = T - offset - 1
                if split_len < 50:
                    break
                if type(data['data']) in [np.ndarray, np.array]:
                    choice_data = data['data'][:split_len, :]
                elif type(data['data']) is list:
                    if i == len(events_dict) - 1:
                        split_onset = random.choice(np.arange(split_len))
                        choice_data = np.concatenate(data['data'], axis=0)[split_onset:split_onset+split_len, :]
                    else:
                        choice_data = random.choice(data['data'])[:split_len, :]
                offset = choice_data.shape[0] + onset - 1
                events_label[onset:offset+1,data['id']] = 1
                events_data[onset:offset+1, :] = choice_data
                onset = offset + 1
                if onset >= T:
                    break
            events_len[count] = onset

        if bg is not None:
            events_data = events_data + bg[:,:T,:]

        return events_data, events_label, events_len

    def _get_sample(self, filename):
        data = np.load(self.data_dir / filename.replace("wav", "npy")).astype(np.float32)
        return data


    def _get_label(self, filename):
        if {"onset", "offset", "event_label"}.issubset(self.df.columns):
            # get strong label
            cols = ["onset", "offset", "event_label"]
            label = self.df[self.df.filename == filename][cols] # dataframe
            if label.empty:
                label = []
        else:
            label = "empty"  # trick to have -1 for unlabeled data and concat them with labeled
            if "filename" not in self.df.columns:
                raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(self.df.columns)
                )
        if self.encode_function is not None:
            # labels are a list of string or list of list [[label, onset, offset]]
            label = self.encode_function(label) # label - (625, 10)
        return label


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
            data, label = self.transforms((data, label))

        # label pooling here because data augmentation may handle label (e.g. time shifting)
        # select center frame as a pooled label
        label = label[self.ptr // 2 :: self.ptr, :]

        # Return twice data with different augmentation if use mean teacher training
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
            # get weak label
            if "event_labels" in self.df.columns:
                label = self.df[self.df.filename == filename]["event_labels"].values[0]
                if pd.isna(label):
                    label = []
                if type(label) is str:
                    if label == "":
                        label = []
                    else:
                        label = label.split(",")
            # get strong label
            else:
                cols = ["onset", "offset", "event_label"]
                label = self.df[self.df.filename == filename][cols]
                if filename in self.offset and label.offset.iloc[-1] > self.offset[filename]:
                    label.offset.iloc[-1] = self.offset[filename]
                # label[]
                if label.empty:
                    label = []
        else:
            label = "empty"  # trick to have -1 for unlabeled data and concat them with labeled
            if "filename" not in self.df.columns:
                raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(self.df.columns)
                )
        if self.encode_function is not None:
            # labels are a list of string or list of list [[label, onset, offset]]
            label = self.encode_function(label)
        return label