from typing import Any


from medmnist import OrganMNIST3D
import matplotlib.pyplot as plt
img_size = 64
train_ds = OrganMNIST3D(split="train", download=True, size=img_size)
val_ds = OrganMNIST3D(split="val", download=True, size=img_size)
test_ds = OrganMNIST3D(split="test", download=True, size=img_size)
n_classes = len(train_ds.info["label"])

print(f"Number of classes: {n_classes}")
print(f"Number of training samples: {len(train_ds)}")
print(f"Number of validation samples: {len(val_ds)}")
print(f"Number of test samples: {len(test_ds)}")

# # Plot grid of images with labels
# fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
# for i, ax in enumerate(axes.flat):
#     img, label = train_ds[i]
#     ax.imshow(img[0, 28 // 2, ...], cmap="gray")
#     ax.set_title(f"Label: {label}")
#     ax.axis("off")
# plt.show()

import torch
import numpy as np

def central_slab(x, axis, width=3):
    center = x.shape[axis] // 2
    start = max(0, center - width // 2)
    index = [slice(None)] * 3
    index[axis] = slice(start, start + width)
    return x[tuple(index)].mean(axis)

def central_slice(x, axis):
    center = x.shape[axis] // 2
    index = [slice(None)] * 3
    index[axis] = slice(center, center + 1)
    return x[tuple(index)].squeeze(axis)

def extract_views(volume, select_method=central_slab):
    x = torch.as_tensor(np.asarray(volume), dtype=torch.float32).squeeze(0) # [V, H, W]
    x = (x - x.mean()) / (x.std() + 1e-10) # [V, H, W]
    axial = select_method(x, axis=0) # [1, H, W]
    coronal = select_method(x, axis=1) # [1, V, W]
    sagittal = select_method(x, axis=2) # [1, V, H]
    return torch.stack([axial, coronal, sagittal]).unsqueeze(1) # [3, 1, H, W]

x = torch.randn(1, 64, 64, 64)
views = extract_views(x)
# print(views.shape)

from torch.utils.data import Dataset

class MultiViewDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        volume, label = self.dataset[idx]
        views = extract_views(volume) # [3, 1, H, W]
        label = torch.as_tensor(label).reshape(-1)[0].long()
        return views, label
    

mv_trainds = MultiViewDataset(train_ds)
bs = 4
from torch.utils.data import DataLoader
train_loader = DataLoader(mv_trainds, batch_size=bs, shuffle=True)
xb, yb = next(iter(train_loader))

# print(xb.shape, yb.shape)
# sampling stats of batch
itr = iter(train_loader)
means, stds = [], []
for _ in range(5):
    xb, yb = next(itr)
    means.append(xb.mean().item())
    stds.append(xb.std().item())
print(f"Mean: {np.mean(means)}, Std: {np.mean(stds)}")

# Visualize a batch of views, taking input as (bs, 3, 1, H, W)
# Plot grid of views of each sample in the batch, each sample should have 3 views in a 3x1 grid
# view_names = ["axial", "coronal", "sagittal"]
# fig, axes = plt.subplots(nrows=3, ncols=bs, figsize=(bs * 2.5, 7.5), squeeze=False)
# for i in range(bs):
#     for v in range(3):
#         ax = axes[v, i]
#         ax.imshow(xb[i, v, 0].numpy(), cmap="gray")
#         label = yb[i].item() if yb[i].numel() == 1 else yb[i].tolist()
#         ax.set_title(f"{view_names[v]} | y={label}" if v == 0 else view_names[v])
#         ax.axis("off")
# plt.tight_layout()
# plt.show()

import torch
import torch.nn as nn
from einops.layers.torch import Reduce


class VisionEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            # layer 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # layer 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # layer 3
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        return self.network(x).flatten(1)

model = VisionEncoder()
xb = torch.randn(bs, 1, 64, 64)
out = model(xb)
print(out.shape)


from einops import rearrange
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy

from odyssey.training.callbacks import TrainCB, default_cbs
from odyssey.training.learner import Learner


class MultiViewClassifier(nn.Module):
    def __init__(self, encoder, n_classes, feature_dim=128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, n_classes)

    def forward(self, x): # (bs, 3, 1, H, W)
        B, V, C, H, W = x.shape
        # simple view, pick axial view; 1 is axial, 0 is coronal, 2 is sagittal
        x = x[:, 2, 0, ...].unsqueeze(1) # (B, 1, H, W)
        return self.head(self.encoder(x))

class LateFusionClassifier(nn.Module):
    def __init__(self, encoder, n_classes, feature_dim=128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim*3, n_classes) # 3 views are concatenated

    def forward(self, x): # (bs, 3, 1, H, W)
        B, V, C, H, W = x.shape
        # print("B:", B, "V:", V, "C:", C, "H:", H, "W:", W)
        x = rearrange(x, "b v c h w -> (b v) c h w") # .squeeze(1)
        # print(x.shape)
        x = self.encoder(x)
        # x = rearrange(x, "(b v) c -> b v c", b=B).mean(dim=1)
        x = rearrange(x, "(b v) c -> b (v c)", b=B)
        # print("output shape:", x.shape)
        return self.head(x)


class GatedFusionClassifier(nn.Module):
    def __init__(self, encoder, n_classes, feature_dim=128):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        # Fuse views by a weighted sum, so the head sees one feature_dim vector.
        self.head = nn.Linear(self.feature_dim, n_classes)
        # Project each view's feature independently to a scalar score.
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x): # (bs, 3, 1, H, W)
        B, V, C, H, W = x.shape # B, V=3, C=1, H, W
        x = rearrange(x, "b v c h w -> (b v) c h w")
        features = self.encoder(x) # (B*V, feature_dim)
        features = rearrange(features, "(b v) c -> b v c", b=B) # (B, V, C)
        scores = self.gate(features) # (B, V, 1)
        softmax_score = F.softmax(scores, dim=1) # softmax over views
        x = (features * softmax_score).sum(dim=1) # (B, C)
        return self.head(x)

class DataLoaders:
    def __init__(self, train, valid):
        self.train, self.valid = train, valid

bs = 128
n_epochs = 50
n_classes = len(train_ds.info["label"])
train_loader = DataLoader(mv_trainds, batch_size=bs, shuffle=True)
val_loader = DataLoader(MultiViewDataset(val_ds), batch_size=bs, shuffle=False)
dls = DataLoaders(train_loader, val_loader)
# model = MultiViewClassifier(VisionEncoder(), n_classes)
# model = LateFusionClassifier(VisionEncoder(), n_classes)
model = GatedFusionClassifier(VisionEncoder(), n_classes)
def loss_func(preds, y):
    return F.cross_entropy(preds, y.view(-1).long())


cbs = default_cbs(
    train=TrainCB(),
    metrics=dict(accuracy=MulticlassAccuracy(num_classes=n_classes)),
    include_compile_cb=False,
)
learn = Learner(model, dls, loss_func, torch.optim.AdamW, lr=1e-3, cbs=cbs)
learn.fit(n_epochs)

