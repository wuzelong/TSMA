import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import repeat, rearrange
from layers.Attn_Bias import BinaryAttentionBias
from layers.Attn_Projection import QueryKeyProjection, RotaryProjection
from utils.masking import TimerMultivariateMask, TimerCovariateMask


class TimeAttention(nn.Module):

    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512,
                 num_heads=8, max_len=100, covariate=False, flash_attention=False, n_vars=7, dim_R=512):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection,
                                           kwargs=dict(max_len=max_len), partial_factor=(0.0, 0.5), )
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)
        self.R_table1 = nn.Embedding(n_vars, dim_R)
        self.R_table2 = nn.Embedding(n_vars, dim_R)

    def forward(self, queries, keys, values, date, attn_mask, n_vars, n_tokens):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:  # False
            values = values.permute(0, 2, 1, 3)

        queries = rearrange(queries, 'b h (c n) e -> b h c n e', c=n_vars)
        keys = rearrange(keys, 'b h (c n) e -> b h c n e', c=n_vars)
        date = repeat(date, 'b n -> b h c n', h=H, c=n_vars)
        queries, keys = self.qk_proj(
            queries, keys, query_id=date, kv_id=date)
        queries = rearrange(queries, 'b h c n e -> b h (c n) e')
        keys = rearrange(keys, 'b h c n e -> b h (c n) e')

        scale = self.scale or 1. / sqrt(E)
        var_id = repeat(torch.arange(n_vars),
                        'C -> (C n_tokens)', n_tokens=n_tokens).to(queries.device)
        var_id_expand = repeat(var_id, 'L -> b h L', b=B, h=1)

        attn_bias = self.attn_bias(var_id_expand, var_id_expand)

        if self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerCovariateMask(
                        B, n_vars, n_tokens, device=queries.device)
                else:
                    attn_mask = TimerMultivariateMask(
                        B, n_vars, n_tokens, device=queries.device)
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask

            A = self.dropout(torch.softmax(scale * scores, dim=-1))

            R1 = self.R_table1.weight
            R2 = self.R_table2.weight

            R1_norm = F.normalize(R1, p=2, dim=-1)
            cos_R1 = torch.matmul(R1_norm, R1_norm.transpose(-2, -1))

            R2_norm = F.normalize(R2, p=2, dim=-1)
            cos_R2 = torch.matmul(R2_norm, R2_norm.transpose(-2, -1))

            upper = torch.triu(cos_R1, diagonal=0)
            lower = torch.tril(cos_R2, diagonal=-1)

            similarity_matrix = upper + lower
            similarity_matrix = similarity_matrix.repeat_interleave(n_tokens, dim=0).repeat_interleave(n_tokens, dim=1)
            A = A * similarity_matrix
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, date, attn_mask=None, n_vars=None, n_tokens=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            date,
            attn_mask,
            n_vars=n_vars,
            n_tokens=n_tokens
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

