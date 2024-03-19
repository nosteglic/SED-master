import warnings

import torch

from models.baseline_model import CNN
from models.conformer.conformer_encoder import ConformerEncoder

class SEDModel(torch.nn.Module):
    def __init__(
        self,
        n_class,
        cnn_kwargs=None,
        encoder_kwargs=None,
        pooling="token",
        layer_init="pytorch",
        use_clean=False,
        num_label=50,
    ):
        super(SEDModel, self).__init__()

        self.n_class = n_class
        self.cnn = CNN(n_in_channel=1, **cnn_kwargs)
        self.input_dim = self.cnn.nb_filters[-1]
        adim = encoder_kwargs["adim"]
        self.pooling = pooling

        self.encoder = ConformerEncoder(self.input_dim, **encoder_kwargs)
        self.classifier = torch.nn.Linear(adim, n_class)

        if use_clean:
            self.projection1 = torch.nn.Linear(adim, adim * 4)
            self.projection2 = torch.nn.Linear(adim * 4, adim * 2)
            self.embedding = torch.nn.Embedding(num_label, adim * 2)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, n_class)
            self.sigmoid = torch.sigmoid
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            self.linear_emb = torch.nn.Linear(1, self.input_dim)
        self.reset_parameters(layer_init)

    def forward(self, x, mask=None, use_clean=False, use_concat=False):
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if freq != 1:
            warnings.warn(
                f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq"
            )
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)

        if self.pooling == "token":
            tag_token = self.linear_emb(torch.ones(x.size(0), 1, 1).cuda())
            x = torch.cat([tag_token, x], dim=1)

        x, _ = self.encoder(x, mask)

        logits = None
        if use_clean:
            if use_concat:
                x_origin = x[: bs // 2, 1:, :]
            else:
                x_origin = x[:, 1:, :]
            x_origin = self.projection1(x_origin)
            x_origin = self.projection2(x_origin)
            logits = torch.cosine_similarity(
                x_origin.unsqueeze(2),
                self.embedding.weight.unsqueeze(0).unsqueeze(0),
                dim=-1
            )

        strong = None
        weak = None
        if self.pooling == "attention":
            strong = self.classifier(x)
            sof = self.dense(x)
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (torch.sigmoid(strong) * sof).sum(1) / sof.sum(1)
            weak = torch.log(weak / (1 - weak))
        elif self.pooling == "token":
            x = self.classifier(x)
            weak = x[:, 0, :]
            strong = x[:, 1:, :]
        elif self.pooling == "auto":
            strong = self.classifier(x)
            weak = self.autopool(strong)

        results = {}
        results["strong"] = strong
        results["weak"] = weak
        if use_concat:
            results["concat"] = x[bs//2:, 1:, :]
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