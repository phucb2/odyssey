"""Model factories and building blocks."""
import math

import torch.nn as nn

def create_model():
    m,nh = 28*28,50
    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
    return model

class Reshape(nn.Module):
    "View flattened batch input as `(bs, *shape)` — keeps dataloader and model input aligned."
    def __init__(self, *shape): super().__init__(); self.shape = shape
    def forward(self, x): return x.view(x.shape[0], *self.shape)
    @property
    def flat_dim(self): return math.prod(self.shape)

def conv(ni, nf, ks=3, act=True):
    return nn.Sequential(nn.Conv2d(ni, nf, ks, padding=ks//2), nn.SiLU(inplace=True) if act else nn.Identity())

def create_cnn_model(nf=32, n_out=10, sz=28, in_ch=1):
    return nn.Sequential(
        Reshape(in_ch, sz, sz),
        conv(in_ch, 8, ks=5),
        conv(8, 16),
        conv(16, 16),
        conv(16, 32),
        conv(32, nf),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(nf, n_out),
    )

class ResBlock(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ni, nf, 3, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1, bias=False),
            nn.BatchNorm2d(nf),
        )
        self.shortcut = nn.Identity() if ni == nf else nn.Sequential(
            nn.Conv2d(ni, nf, 1, bias=False),
            nn.BatchNorm2d(nf),
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


def create_res_model(sz=28, nf=32, n_out=10, *, wide=False, width_mult=2.0, in_ch=1):
    "ResNet for flat `(bs, in_ch*sz*sz)` or channel-first GPU batches."
    if not wide:
        c1, c2, mid_ch, c_pen, nf_out, n_mid = 8, 16, 16, 32, nf, 51
    else:
        wm = float(width_mult)
        c1, c2 = max(4, int(round(8 * wm))), max(8, int(round(16 * wm)))
        mid_ch, c_pen = c2, max(16, int(round(32 * wm)))
        nf_out = max(16, int(round(nf * wm)))
        n_mid = {1.5: 18, 2.0: 7, 2.5: 2}.get(wm, max(1, int(round(51 / wm ** 2))))
    layers = [Reshape(in_ch, sz, sz), ResBlock(in_ch, c1), ResBlock(c1, c2)]
    layers.extend([ResBlock(mid_ch, mid_ch) for _ in range(n_mid)])
    layers += [
        ResBlock(mid_ch, c_pen),
        ResBlock(c_pen, nf_out),
        nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(nf_out, n_out)),
    ]
    return nn.Sequential(*layers)
