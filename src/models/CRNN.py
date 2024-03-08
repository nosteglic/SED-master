import warnings

import torch
import torch.nn as nn
from models.baseline_model import CNN, BiGRU
from models.concat_data import concat_data1, calculate_similarity

class SEDModel(nn.Module):
    def __init__(self,
                 n_class,
                 attention=True,
                 gen_count=2,
                 ptr=4,
                 cnn_kwargs=None,
                 rnn_kwargs=None,):
        super(SEDModel, self).__init__()

        self.n_class = n_class
        self.ptr = ptr
        self.gen_count=gen_count
        self.cnn = CNN(n_in_channel=1, **cnn_kwargs)
        self.input_dim = self.cnn.nb_filters[-1]
        self.rnn = BiGRU(n_in=self.input_dim, **rnn_kwargs)
        self.attention = attention

        self.dropout = nn.Dropout(cnn_kwargs["conv_dropout"])
        n_hidden = rnn_kwargs["n_hidden"]
        self.classifier = nn.Linear(n_hidden * 2, n_class)

        if self.attention:
            self.dense = nn.Linear(n_hidden * 2, n_class)
            self.softmax = nn.Softmax(dim=-1)

        self.reset_parameters()

    def forward(self, x, events=None, bg=None): # x (bs, chan, frames, freqs) - (12, 1, 625, 128)
        x_batch = x.shape[0]
        if events is not None:
            events_x, events_y, events_len = concat_data1(
                events,
                n_class=self.n_class,
                gen_count=self.gen_count,
                T=x.shape[-2],
                F=x.shape[-1],
                ptr=self.ptr,
                bg=bg,
            )

            x = torch.cat([x, events_x], dim=0)

        x = self.cnn(x) # x - (12, 128, 156, 1)
        bs, chan, frames, freq = x.size()
        if freq != 1:
            warnings.warn(
                f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq"
            )
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1) # x - (12, 128, 156)
            x = x.permute(0, 2, 1) # x - [bs, frames, chan] - (12, 156, 128)

        #rnn
        x = self.rnn(x) #x - [bs, frames, 2 * chan] - (12, 156, 256)
        x = self.dropout(x)

        #classifier
        strong = self.classifier(x) #strong size : [bs, frames, n_class] - (12, 156, 10)
        if self.attention:
            sof = self.dense(x) #sof size : [bs, frames, n_class] - (12, 156, 10)
            sof = self.softmax(sof) #sof size : [bs, frames, n_class]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (torch.sigmoid(strong) * sof).sum(1) / sof.sum(1) # [bs, n_class] - (12, 10)
            weak = torch.log(weak / (1 - weak))
        else:
            weak = strong.mean(1)

        if events is not None:
            return {
                "strong": strong,
                "weak": weak,
                "events_x": calculate_similarity(x[x_batch:,:,:]),
                "events_y": events_y,
                "events_len": events_len
            }
        return {"strong": strong, "weak": weak}

    def reset_parameters(self, initialization: str = "pytorch"):
        if initialization.lower() == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if initialization.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif initialization.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif initialization.lower() == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif initialization.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown initialization: {initialization}")
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()