"""Matched A26 CNN residual in physical material coordinates.

The caller normalizes the twelve input channels with the pooled TRAIN
statistics.  A single TRAIN material RMS scales both output channels for all
three methods.  ``forward`` and ``predict_mass`` return real mass coordinates;
``predict_chi`` returns complex physical chi.  There is no physics operation in
the decoder graph.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from a25.torch_physics import mass_unpack


def project_mass(mass, volume, *, real_lower=-.5, real_upper=4.,
                 imag_lower=0., imag_upper=.5):
    """Apply the common full-grid material box without a weak-space projection."""
    volume = float(volume)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("POSITIVE_FINITE_CELL_VOLUME_REQUIRED")
    if mass.is_complex() or mass.ndim != 2 or mass.shape[-1] % 2:
        raise ValueError("BATCHED_REAL_MATERIAL_MASS_REQUIRED")
    n, scale = mass.shape[-1] // 2, math.sqrt(volume)
    return torch.cat((mass[:, :n].clamp(real_lower * scale, real_upper * scale),
                      mass[:, n:].clamp(imag_lower * scale, imag_upper * scale)), dim=1)


class ResidualDecoder(nn.Module):
    """Five convolutions, dilations 1/2/4/2/1, and a zero output head.

    Width 32 has 31,810 trainable parameters.  Width 16 is the only frozen
    budget fallback.  The unconstrained CNN residual is multiplied by the
    shared scalar physical material RMS and sqrt(cell volume), then added to
    the physical base and projected into the common material box.
    """
    dilations = (1, 2, 4, 2, 1)

    def __init__(self, width=32, *, real_lower=-.5, real_upper=4.,
                 imag_lower=0., imag_upper=.5):
        super().__init__()
        if type(width) is not int or width not in {16, 32}:
            raise ValueError("A26_DECODER_WIDTH_MUST_BE_32_OR_16")
        self.width = width
        self.bounds = {"real_lower": float(real_lower), "real_upper": float(real_upper),
                       "imag_lower": float(imag_lower), "imag_upper": float(imag_upper)}
        if (not all(math.isfinite(x) for x in self.bounds.values())
                or real_lower >= real_upper or imag_lower >= imag_upper):
            raise ValueError("FINITE_ORDERED_MATERIAL_BOX_REQUIRED")
        channels = (12, width, width, width, width, 2)
        layers = []
        for index, dilation in enumerate(self.dilations):
            layers.append(nn.Conv2d(channels[index], channels[index + 1], 3,
                                    padding=dilation, dilation=dilation))
            if index < len(self.dilations) - 1:
                layers.append(nn.GELU())
        self.body = nn.Sequential(*layers)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, normalized_features, base_mass, material_scale, volume=1. / 576):
        """Return float64 [B,2*grid**2] mass from normalized float32 features."""
        if (not isinstance(normalized_features, torch.Tensor)
                or normalized_features.ndim != 4 or normalized_features.shape[1] != 12
                or normalized_features.shape[-2] != normalized_features.shape[-1]
                or normalized_features.dtype != torch.float32):
            raise ValueError("NORMALIZED_FLOAT32_TWELVE_CHANNEL_FEATURES_REQUIRED")
        batch, _, height, width = normalized_features.shape
        if (not isinstance(base_mass, torch.Tensor) or base_mass.dtype != torch.float64
                or base_mass.shape != (batch, 2 * height * width)
                or base_mass.device != normalized_features.device):
            raise ValueError("MATCHED_FLOAT64_BASE_MASS_REQUIRED")
        if not torch.isfinite(normalized_features).all() or not torch.isfinite(base_mass).all():
            raise ValueError("FINITE_DECODER_INPUTS_REQUIRED")
        volume = float(volume)
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("POSITIVE_FINITE_CELL_VOLUME_REQUIRED")
        scale = torch.as_tensor(material_scale, dtype=torch.float64,
                                device=normalized_features.device)
        if scale.numel() != 1 or not torch.isfinite(scale).all() or float(scale) <= 0:
            raise ValueError("POSITIVE_SHARED_SCALAR_MATERIAL_RMS_REQUIRED")
        residual = self.body(normalized_features).flatten(1).to(torch.float64)
        mass = base_mass + residual * scale.detach().reshape(()) * math.sqrt(volume)
        return project_mass(mass, volume, **self.bounds)

    def predict_mass(self, normalized_features, base_mass, material_scale, volume=1. / 576):
        return self.forward(normalized_features, base_mass, material_scale, volume)

    def predict_chi(self, normalized_features, base_mass, material_scale, volume=1. / 576):
        return mass_unpack(self.forward(normalized_features, base_mass, material_scale, volume), volume)
