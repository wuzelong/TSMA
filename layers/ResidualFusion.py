import torch
from torch import nn


class ResidualFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, embed_out1, embed_outA, embed_outB):
        concat_features = torch.cat([embed_out1, embed_outA, embed_outB], dim=-1)
        fused_residual = self.scale_fusion(concat_features)
        final_features = embed_out1 + fused_residual
        return final_features