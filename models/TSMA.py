import torch
from torch import nn

from layers.ResidualFusion import ResidualFusion
from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_token_len = configs.input_token_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)
        self.output_attention = configs.output_attention
        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(True, attention_dropout=configs.dropout,
                                    output_attention=self.output_attention,
                                    d_model=configs.d_model, num_heads=configs.n_heads,
                                    covariate=configs.covariate, flash_attention=configs.flash_attention, n_vars=configs.n_vars, dim_R=configs.dim_R),
                                    configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.head = nn.Linear(configs.d_model, configs.output_token_len)  # 输出头
        self.use_norm = configs.use_norm
        self.fuse_layer = ResidualFusion(configs.d_model)

        self.output_token_len = configs.output_token_len
        self.d_model = configs.d_model


    def forecast(self, x, batch_date, x_mark, y_mark):
        B, S, C = x.shape

        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        x = x.permute(0, 2, 1)

        x1 = x[..., -S//2:]
        batch_date1 = batch_date[..., -S//2:]
        x1 = x1.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x1.shape[2]
        batch_date1 = batch_date1.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        batch_date1 = batch_date1[:, :, 0:1]
        batch_date1 = batch_date1.squeeze(-1)
        embed_out1 = self.embedding(x1)
        embed_out1 = embed_out1.reshape(B, C * N, -1)
        embed_out1, attns = self.blocks(embed_out1, batch_date1, n_vars=C, n_tokens=N)

        xA = x[..., ::2].clone()
        batch_dateA = batch_date[..., ::2].clone()
        xA = xA.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        batch_dateA = batch_dateA.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        batch_dateA = batch_dateA[..., 0:1]
        batch_dateA = batch_dateA.squeeze(-1)
        embed_outA = self.embedding(xA)
        embed_outA = embed_outA.reshape(B, C * N, -1)
        embed_outA, _ = self.blocks(embed_outA, batch_dateA, n_vars=C, n_tokens=N)

        xB = x[..., 1::2].clone()
        batch_dateB = batch_date[..., 1::2].clone()
        xB = xB.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        batch_dateB = batch_dateB.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        batch_dateB = batch_dateB[..., 0:1]
        batch_dateB = batch_dateB.squeeze(-1)
        embed_outB = self.embedding(xB)
        embed_outB = embed_outB.reshape(B, C * N, -1)
        embed_outB, _ = self.blocks(embed_outB, batch_dateB, n_vars=C, n_tokens=N)

        embed_out = self.fuse_layer(embed_out1, embed_outA, embed_outB)

        dec_out = self.head(embed_out)
        dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)
        dec_out = dec_out.permute(0, 2, 1)


        if self.use_norm:
            dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def forward(self, x, batch_date, x_mark, y_mark):
        return self.forecast(x, batch_date, x_mark, y_mark)


