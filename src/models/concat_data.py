import torch
import random
import numpy as np

def calculate_similarity(M):
    M = M / torch.linalg.vector_norm(M, dim=2, keepdim=True)
    # return torch.sigmoid(torch.bmm(M, M.permute(0,2,1)))
    return torch.bmm(M, M.permute(0,2,1))

def concat_data1(events_batch, n_class, bg=None, gen_count=2, T=625, F=128, ptr=4):
    """
    将x拼成两条拼接音频，并生成这两条音频的label
    :param events_batch: 字典列表，长度为batch_size
        一个dict包括id和events列表，前者标志合成音频id，后者为对应使用的事件们
        events: 字典列表，长度为对应的合成音频的事件长度
            一个dict包括事件标签event_id、特征 data
                特征data的size为(T, F)，其中T不固定，F在CRNN中为128
    :param gen_count: 生成的拼接音频的数量
    :return: 拼接音频以及对应的标签
    """
    batch_size = len(events_batch)
    gen_label = torch.zeros((gen_count*batch_size, T, n_class))
    gen_data = torch.zeros((gen_count*batch_size, T, F))
    gen_len = [0] * gen_count*batch_size
    for count in range(gen_count):
        for j, events in enumerate(events_batch):
            random.shuffle(events["events"])
            onset = 0
            offset = 0
            for i, data in enumerate(events["events"]):
                split_len = T - offset - 1
                if split_len < 50:
                    break
                if type(data['data']) is torch.Tensor:
                    choice_data = data['data'][:split_len, :]
                elif type(data['data']) is list:
                    if i == len(events['events'])-1:
                        split_onset = random.choice(np.arange(split_len))
                        choice_data = torch.cat(data['data'])[split_onset:split_onset+split_len, :]
                    else:
                        choice_data = random.choice(data['data'])[:split_len, :]
                offset = choice_data.shape[0] + onset - 1
                gen_label[count*batch_size+j, onset:offset + 1, data['event_id']] = 1
                gen_data[count*batch_size+j, onset:offset+1, :] = choice_data
                onset = offset + 1
                if onset >= T:
                    break
            gen_len[count*batch_size+j] = onset

    gen_label = gen_label[:, ptr // 2:: ptr, :]

    gen_len = [(i - ptr//2) // ptr + 1 for i in gen_len]

    if bg is not None:
        gen_data = gen_data.add(bg[:,:T,:])

    gen_label = gen_label.to("cuda" if torch.cuda.is_available() else "cpu")
    gen_data = gen_data.to("cuda" if torch.cuda.is_available() else "cpu")

    return (torch.unsqueeze(gen_data, dim=1), gen_label, gen_len)

def concat_data2(self, x):
    """
    既输入原本的x，也输入拼接的x
    :param x:
    :return:
    """

    pass