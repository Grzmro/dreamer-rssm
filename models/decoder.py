"""Transposed-CNN decoder reconstructing observations from RSSM features.

NOTE: This is a simplified observation model relative to DreamerV3. We
predict only the mean of a unit-variance diagonal Gaussian and train with
MSE, which equals the Gaussian negative log-likelihood up to a constant.
DreamerV3 additionally uses symlog-transformed targets; not needed for
[-0.5, 0.5]-normalized pixels at this phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    """Decode [B, feat_dim] features ([h_t, z_t]) to [B, C, 64, 64] means.

    Mirror of :class:`~models.encoder.ConvEncoder`: linear projection to the
    deepest feature map, then 4 stride-2 transposed convolutions up to 64x64.
    The final layer has no norm/activation — outputs are unbounded means
    matched against targets in [-0.5, 0.5] via MSE.
    """

    def __init__(
        self,
        feat_dim: int,
        out_channels: int = 3,
        depth: int = 32,
        num_layers: int = 4,
        norm: bool = True,
    ):
        super().__init__()
        self._start_res = 64 // 2**num_layers
        self._start_ch = depth * 2 ** (num_layers - 1)
        self.fc = nn.Linear(feat_dim, self._start_ch * self._start_res**2)

        layers: list[nn.Module] = []
        ch_in = self._start_ch
        for i in reversed(range(num_layers)):
            last = i == 0
            ch_out = out_channels if last else depth * 2 ** (i - 1)
            layers.append(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)
            )
            if not last:
                if norm:
                    layers.append(nn.GroupNorm(1, ch_out))
                layers.append(nn.SiLU())
            ch_in = ch_out
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(feat)
        x = x.view(-1, self._start_ch, self._start_res, self._start_res)
        return self.net(x)
