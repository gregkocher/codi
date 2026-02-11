# ABOUTME: Interpolation utilities for latent vectors (lerp and slerp).
# ABOUTME: Used by interpolation experiments to smoothly transition between
# ABOUTME: two latent reasoning vectors from different prompts.

import torch


def lerp(v1: torch.Tensor, v2: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Linear interpolation between two tensors.

    v = (1 - alpha) * v1 + alpha * v2

    Args:
        v1: Source tensor of any shape.
        v2: Target tensor (same shape as v1).
        alpha: Interpolation weight. 0.0 = v1, 1.0 = v2.

    Returns:
        Interpolated tensor with the same shape.
    """
    return (1.0 - alpha) * v1 + alpha * v2


def slerp(v1: torch.Tensor, v2: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.

    Moves along the great circle on the hypersphere, preserving
    the norm throughout the interpolation path. Falls back to lerp
    when the vectors are nearly parallel (angle < 1e-6).

    Args:
        v1: Source tensor of shape (..., hidden_dim).
        v2: Target tensor (same shape as v1).
        alpha: Interpolation weight. 0.0 = v1, 1.0 = v2.

    Returns:
        Interpolated tensor with the same shape and approximately
        the same norm as the inputs.
    """
    # Flatten to 2D for the computation, restore shape at the end
    original_shape = v1.shape
    v1_flat = v1.reshape(-1, v1.shape[-1]).float()
    v2_flat = v2.reshape(-1, v2.shape[-1]).float()

    # Normalize to unit vectors
    v1_norm = v1_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v2_norm = v2_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v1_unit = v1_flat / v1_norm
    v2_unit = v2_flat / v2_norm

    # Cosine of the angle between the vectors
    cos_angle = (v1_unit * v2_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)

    # If angle is tiny, fall back to lerp (avoid division by zero in sin)
    sin_angle = torch.sin(angle)
    use_lerp = sin_angle.abs() < 1e-6

    # Slerp formula: sin((1-t)*θ)/sin(θ) * v1 + sin(t*θ)/sin(θ) * v2
    safe_sin = sin_angle.clamp(min=1e-8)  # for the division
    w1 = torch.sin((1.0 - alpha) * angle) / safe_sin
    w2 = torch.sin(alpha * angle) / safe_sin

    # Fall back to lerp where vectors are nearly parallel
    w1 = torch.where(use_lerp, torch.tensor(1.0 - alpha, device=v1.device), w1)
    w2 = torch.where(use_lerp, torch.tensor(alpha, device=v1.device), w2)

    result = w1 * v1_flat + w2 * v2_flat

    # Interpolate the norms linearly and rescale
    target_norm = (1.0 - alpha) * v1_norm + alpha * v2_norm
    result_norm = result.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    result = result * (target_norm / result_norm)

    return result.to(v1.dtype).reshape(original_shape)
