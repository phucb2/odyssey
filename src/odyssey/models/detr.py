"""Small educational DETR with a pretrained ResNet18 backbone."""
import math

import torch
import torch.nn as nn
import torchvision.models as models


class PositionalEncoding2D(nn.Module):
    """Sine/cosine positional encoding for a 2D feature map (DETR-style)."""

    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y = torch.arange(h, device=x.device, dtype=x.dtype).view(h, 1).repeat(1, w)
        x_pos = torch.arange(w, device=x.device, dtype=x.dtype).view(1, w).repeat(h, 1)
        y = y / max(h - 1, 1)
        x_pos = x_pos / max(w - 1, 1)

        dim = self.d_model // 2
        div = torch.exp(
            torch.arange(0, dim, 2, device=x.device, dtype=x.dtype)
            * (-math.log(self.temperature) / dim)
        )

        pe_y = torch.zeros(h, w, dim, device=x.device, dtype=x.dtype)
        pe_x = torch.zeros(h, w, dim, device=x.device, dtype=x.dtype)
        pe_y[:, :, 0::2] = torch.sin(y.unsqueeze(-1) * div)
        pe_y[:, :, 1::2] = torch.cos(y.unsqueeze(-1) * div)
        pe_x[:, :, 0::2] = torch.sin(x_pos.unsqueeze(-1) * div)
        pe_x[:, :, 1::2] = torch.cos(x_pos.unsqueeze(-1) * div)
        pe = torch.cat((pe_y, pe_x), dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        return pe


class MLP(nn.Module):
    """Simple multi-layer perceptron for bbox regression."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_features = in_dim if i == 0 else hidden_dim
            out_features = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallDETR(nn.Module):
    """Educational DETR: ImageNet ResNet18 backbone + random transformer/heads."""

    def __init__(
        self,
        *,
        num_classes: int = 80,
        num_queries: int = 50,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.input_proj = nn.Conv2d(512, d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding2D(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # images: (B, 3, H, W)
        features = self.backbone(images)
        pos = self.pos_encoder(features)
        src = self.input_proj(features) + pos

        b, c, h, w = src.shape
        src = src.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        query = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        hs = self.transformer(src, query)  # (B, num_queries, C)

        logits = self.class_head(hs)
        boxes = self.bbox_head(hs).sigmoid()
        return {"pred_logits": logits, "pred_boxes": boxes}


def create_detr_model(
    *,
    num_classes: int = 80,
    num_queries: int = 50,
    d_model: int = 256,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    freeze_backbone: bool = True,
) -> SmallDETR:
    return SmallDETR(
        num_classes=num_classes,
        num_queries=num_queries,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        freeze_backbone=freeze_backbone,
    )
