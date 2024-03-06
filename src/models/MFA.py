import torch

from models.baseline_model import CNN
from models.conformer.conformer_encoder_cat import ConformerEncoder

class SEDModel(torch.nn.Module):
    def __init__(
        self,
        n_class,
        cnn_kwargs=None,
        encoder_kwargs=None,
        pooling="token",
        layer_init="pytorch",
    ):
        super(SEDModel, self).__init__()

        self.cnn = CNN(n_in_channel=1, **cnn_kwargs)
        input_dim = self.cnn.nb_filters[-1]
        adim = encoder_kwargs["adim"]
        elayers = encoder_kwargs["elayers"]
        self.pooling = pooling

        self.encoder = ConformerEncoder(input_dim, **encoder_kwargs)

        self.classifier = torch.nn.Linear(adim * elayers, n_class)

        if self.pooling == "attention":
            self.aggregator = torch.nn.AdaptiveAvgPool1d(1)

        elif self.pooling == "token":
            self.linear_emb = torch.nn.Linear(1, input_dim)

        self.reset_parameters(layer_init)

    def forward(self, x, mask=None):

        # input
        x = self.cnn(x)
        x = x.squeeze(-1).permute(0, 2, 1)

        if self.pooling == "token":
            tag_token = self.linear_emb(torch.ones(x.size(0), 1, 1).cuda())
            x = torch.cat([tag_token, x], dim=1)
            x, _ = self.encoder(x, mask)
        elif self.pooling == "attention":
            x, _ = self.encoder(x, mask)
            tag_token = self.aggregator(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = torch.cat([tag_token, x], dim=1)

        x = self.classifier(x)
        weak = x[:, 0, :]
        strong = x[:, 1:, :]

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