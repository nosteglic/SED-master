import warnings

import torch
import torch.nn as nn
from models.baseline_model import CNN, BiGRU

class SEDModel(nn.Module):
    def __init__(self,
                 n_class,
                 attention=True,
                 cnn_kwargs=None,
                 rnn_kwargs=None,
                 use_clean=False,
                 num_label=50,
                 ):
        super(SEDModel, self).__init__()

        self.n_class = n_class
        self.cnn = CNN(n_in_channel=1, **cnn_kwargs)
        self.input_dim = self.cnn.nb_filters[-1]
        self.rnn = BiGRU(n_in=self.input_dim, **rnn_kwargs)
        self.attention = attention

        self.dropout = nn.Dropout(cnn_kwargs["conv_dropout"])
        n_hidden = rnn_kwargs["n_hidden"]
        self.classifier = nn.Linear(n_hidden * 2, n_class)

        if use_clean:
            self.projection1 = torch.nn.Linear(n_hidden * 2, n_hidden * 4)
            self.projection2 = torch.nn.Linear(n_hidden * 4, n_hidden * 2)
            self.embedding = torch.nn.Embedding(num_label, n_hidden * 2)

        if self.attention:
            self.dense = nn.Linear(n_hidden * 2, n_class)
            self.softmax = nn.Softmax(dim=-1)

        self.reset_parameters()

    def forward(self, x, use_clean=False, use_concat=False): # x (bs, chan, frames, freqs) - (12, 1, 625, 128)
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

        logits = None
        if use_clean:
            if use_concat:
                x_origin = x[: bs // 2, :, :]
            else:
                x_origin = x[:, :, :]
            x_origin = self.projection1(x_origin)
            x_origin = self.projection2(x_origin)
            logits = torch.cosine_similarity(
                x_origin.unsqueeze(2),
                self.embedding.weight.unsqueeze(0).unsqueeze(0),
                dim=-1
            )

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

        results = {}
        results["strong"] = strong
        results["weak"] = weak
        if use_concat:
            results["concat"] = x[bs // 2:, :, :]
        if use_clean:
            results['clean'] = logits
        return results

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