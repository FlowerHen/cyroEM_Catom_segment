from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HeatmapLoss(nn.Module):
    def __init__(
        self,
        *,
        pos_weight: float,
        bce_weight: float,
        focal_weight: float,
        dice_weight: float,
        focal_alpha: float,
        focal_gamma: float,
    ) -> None:
        super().__init__()
        if pos_weight <= 0:
            raise ValueError("pos_weight must be positive")
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.bce_weight = float(bce_weight)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

    def forward(
        self, logits: torch.Tensor, heatmap: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.shape != heatmap.shape:
            raise ValueError("logits and heatmap must have identical shapes")
        components: dict[str, torch.Tensor] = {}
        total = logits.new_zeros(())

        if self.bce_weight:
            bce = F.binary_cross_entropy_with_logits(logits, heatmap, pos_weight=self.pos_weight)
            components["bce"] = bce
            total = total + self.bce_weight * bce

        probabilities = torch.sigmoid(logits.float())
        if self.focal_weight:
            element_bce = F.binary_cross_entropy_with_logits(
                logits.float(), heatmap, reduction="none"
            )
            p_t = probabilities * heatmap + (1.0 - probabilities) * (1.0 - heatmap)
            alpha_t = self.focal_alpha * heatmap + (1.0 - self.focal_alpha) * (1.0 - heatmap)
            focal = (alpha_t * (1.0 - p_t).pow(self.focal_gamma) * element_bce).mean()
            components["focal"] = focal
            total = total + self.focal_weight * focal

        if self.dice_weight:
            dimensions = tuple(range(2, probabilities.ndim))
            intersection = (probabilities * heatmap).sum(dim=dimensions)
            denominator = probabilities.sum(dim=dimensions) + heatmap.sum(dim=dimensions)
            dice = (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
            components["dice"] = dice
            total = total + self.dice_weight * dice

        components["total"] = total
        return total, components
