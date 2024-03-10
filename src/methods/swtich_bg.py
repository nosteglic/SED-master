""" Utility classes and functions related to Background-Domain-Switch (Interspeech 2023).
Copyright (c) 2023 Robert Bosch GmbH
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch
import numpy as np

def real_pseudo_strong_labeling(real_data, model, event_threshold, stochastic_iter=1, norm=None, scale=None):
    """Pseudo strong labeling for weak-labeled or unlabeled data based on the trained SED model.
    The model is expected to have strong label predictions as the first output,
    e.g., strong, weak, ... = model(data)
    input shape of data= [batch, dim, time]

    Returns:
        numpy.Array of binary strong labels, shape=[batch, class, time]
    """
    # feature pre-processing functions if any
    pseudo_strong = []
    if scale:
        real_data = scale(real_data)  # e.g., log scale
    if norm:
        real_data = norm(real_data)  # e.g., z-norm
    for _ in range(stochastic_iter):
        pseudo_strong.append(model(real_data)["strong"])  # gets the strong predictions
    pseudo_strong = torch.mean(torch.stack(pseudo_strong), dim=0)
    pseudo_strong = pseudo_strong.cpu().detach().numpy()
    # binarize labels based on the defined threshold
    pseudo_strong[pseudo_strong >= event_threshold] = 1
    pseudo_strong[pseudo_strong < event_threshold] = 0
    return pseudo_strong.astype('int')


def get_bgs_idx(strong_label, min_frames, upsample=1):
    """Parsing background segments that contain zero-target events based on strong labels.
    input shape of strong label= [batch, class, time]

    Returns:
        list of segment starting and ending indexes
    """
    event_detected = np.sum(strong_label, axis=1)
    NoEvent_SegIdx = []
    for i in range(len(event_detected)):
        zero_event_idxs = np.where(event_detected[i] == 0)[0]
        if len(zero_event_idxs) != 0:
            # find segmentation points
            _diff = np.where(np.diff(zero_event_idxs) != 1)[0] + 1
            if 0 not in _diff:
                _diff = np.insert(_diff, 0, 0)
            # output zero-event slices
            zero_event_segment = []
            for j in range(len(_diff)):
                try:
                    seg_start_idx = _diff[j]
                    seg_end_idx = _diff[j + 1]
                    zero_event_segment.append(zero_event_idxs[seg_start_idx:seg_end_idx])
                except:
                    zero_event_segment.append(zero_event_idxs[_diff[j]:])
            zero_event_segment_Idx = []
            for j in range(len(zero_event_segment)):
                # upsampling back to the original length
                seg_start_idx = zero_event_segment[j][0] * upsample
                seg_end_idx = seg_start_idx + len(zero_event_segment[j]) * upsample
                # checking if the segment is long enough
                if (seg_end_idx - seg_start_idx) > min_frames:
                    zero_event_segment_Idx.append([seg_start_idx, seg_end_idx])
            NoEvent_SegIdx.append(zero_event_segment_Idx)
        else:
            NoEvent_SegIdx.append([])
    return NoEvent_SegIdx


def switch_bg(data, domain_seg_idxs, background_seg_pool):
    """Switch the background segments of source domain data (i.e., data & domain_seg) to randomly
    selected backgrounds from another domain's background pool.
    input shape of data= [batch, dim, time]

    Returns:
        torch.Tensor, same shape as the input data after in-place switch operation
    """
    for i in range(len(data)): # 第i条数据[dim, time]
        if len(domain_seg_idxs[i]) != 0:
            for j in range(len(domain_seg_idxs[i])):
                _seg_start_idx = domain_seg_idxs[i][j][0]
                _seg_end_idx = domain_seg_idxs[i][j][1]

                rdn_pool_idx = np.random.choice(len(background_seg_pool), size=1, replace=True)
                target_back = background_seg_pool[int(rdn_pool_idx)]

                # shorter background case: repeat & crop
                if data[i, :, _seg_start_idx:_seg_end_idx].size()[-1] > target_back.size()[-1]:
                    repeat_times = int(
                        np.ceil(data[i, :, _seg_start_idx:_seg_end_idx].size()[-1] / target_back.size()[-1]))
                    target_back = target_back.repeat(1, repeat_times)  # repeat & crop
                    target_back = target_back[:, 0:_seg_end_idx - _seg_start_idx]

                # longer background case: directly crop
                elif data[i, :, _seg_start_idx:_seg_end_idx].size()[-1] < target_back.size()[-1]:
                    target_back = target_back[:, 0:_seg_end_idx - _seg_start_idx]

                # switch background
                data[i, :, _seg_start_idx:_seg_end_idx] = target_back
    return data