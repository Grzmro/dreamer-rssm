"""CNN encoder mapping 64x64 image observations to a flat embedding.

DreamerV2/V3-style: 4 stride-2 convolutions with doubling channel counts,
channel-wise LayerNorm (implemented as GroupNorm with one group) applied
after each convolution and before the activation, SiLU activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """Encode [B, C, 64, 64] observations in [-0.5, 0.5] to [B, embed_dim].

    Args:
        in_channels: observation channels (3 for RGB, 1 for grayscale).
        depth: base channel count; layer i uses ``depth * 2**i`` channels.
        num_layers: number of stride-2 conv layers (64x64 -> 4x4 for 4 layers).
        norm: apply channel-wise LayerNorm after each conv.
    """

    def __init__(
        self,
        in_channels: int = 3,
        depth: int = 32,
        num_layers: int = 4,
        norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = in_channels
        for i in range(num_layers):
            ch_out = depth * 2**i
            layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1))
            if norm:
                layers.append(nn.GroupNorm(1, ch_out))  # channel-wise LayerNorm
            layers.append(nn.SiLU())
            ch_in = ch_out
        self.net = nn.Sequential(*layers)

        final_res = 64 // 2**num_layers
        self.embed_dim = ch_in * final_res * final_res

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """[B, C, 64, 64] -> [B, embed_dim] (no projection; raw flatten)."""
        return self.net(obs).flatten(start_dim=1)
