from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import math

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
        event_dir: Path = None,
        classes: list = None,
    ):
        self.df = df
        self.data_dir = data_dir
        self.encode_function = encode_function
        self.ptr = pooling_time_ratio
        self.transforms = transforms
        self.filenames = df.filename.drop_duplicates().values
        self.twice_data = twice_data

        self.use_events = use_events
        self.event_dir = event_dir

        if type(classes) in [np.ndarray, np.array]:
            self.classes = classes.tolist()

        self._check_exist()


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        data_id = self.filenames[index]
        if self.use_events:
            data, events_list = self._get_sample(data_id)
            events_dict = {"id": data_id, "events": events_list}
        else:
            data = self._get_sample(data_id)
        label = self._get_label(data_id) # label - (625, 10)
        if self.transforms is not None:
            data, label = self.transforms((data, label))
            # data - 0 - (625, 128)
            # data - 1 - (625, 128)
            # label - (625, 10)

        # label pooling here because data augmentation may handle label (e.g. time shifting)
        # select center frame as a pooled label
        label = label[self.ptr // 2 :: self.ptr, :] # label - (156, 10)

        # Return twice data with different augmentation if use mean teacher training
        if not self.twice_data:
            if not self.use_events:
                return (
                    torch.from_numpy(data).float().unsqueeze(0),
                    torch.from_numpy(label).float(),
                    data_id,
                )
            else:
                return (
                    torch.from_numpy(data).float().unsqueeze(0),
                    torch.from_numpy(label).float(),
                    data_id,
                ), events_dict
        else:
            if not self.use_events:
                return (
                    torch.from_numpy(data[0]).float().unsqueeze(0),
                    torch.from_numpy(data[1]).float().unsqueeze(0),
                    torch.from_numpy(label).float(),
                    data_id,
                )
            else:
                return (
                    torch.from_numpy(data[0]).float().unsqueeze(0),
                    torch.from_numpy(data[1]).float().unsqueeze(0),
                    torch.from_numpy(label).float(),
                    data_id,
                ), events_dict

    def _check_exist(self):
        del_ids = []
        for i in range(len(self.filenames)):
            f = self.filenames[i]
            if not os.path.exists(self.data_dir / f.replace("wav", "npy")):
                del_ids.append(i)
        self.filenames = np.delete(self.filenames, del_ids)

    def _get_sample(self, filename):
        data = np.load(self.data_dir / filename.replace("wav", "npy")).astype(np.float32)
        fileid = filename[:-4]
        events_root = f"{self.event_dir}/{fileid}"
        if self.use_events and os.path.exists(events_root):
            events_list = []
            for event_path in os.listdir(events_root):
                if event_path[-4:] == ".npy":
                    event_data = np.load(f"{events_root}/{event_path}").astype(np.float32)
                    event_data = torch.from_numpy(event_data).float()
                    if event_data.shape[0] > data.shape[0] // 2:
                        event_data = torch.chunk(
                            input=event_data,
                            chunks=math.ceil(event_data.shape[0] * 2 / data.shape[0]),
                            dim=0,
                        )
                    event_name = "_".join(event_path.split("_")[1:])[:-4]
                    ind = self.classes.index(event_name)
                    events_list.append(
                        {
                            # "filename": event_path.replace("npy", "wav"),
                             # "event_label": event_name,
                             "event_id": ind,
                             "data": event_data,
                        },
                    )
                    # events_df = pd.DataFrame.from_dict(events_data_list)
                elif event_path[-4:] == ".tsv":
                    pass
                    # events_data_df= pd.DataFrame.from_dict(events_data_list)
                    # event_duration_df = pd.read_csv(events_root / event_path, sep="\t")
                    # events_df = pd.merge(events_data_df, event_duration_df, on="filename")
                    #
                    # events_df.duration = (events_df.duration * self.n_frames_per_sec).astype(int)
            return data, events_list
        else:
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
        # temp = self.filenames
        # self.filenames = np.delete(self.filenames, del_ids)
        # missing_files = np.setdiff1d(temp, self.filenames)
        # missing_unlabel = pd.DataFrame(data=missing_files, index=None, columns=['filename'])
        # missing_unlabel.to_csv('/data2/syx/missing_validation.csv', index=False)


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