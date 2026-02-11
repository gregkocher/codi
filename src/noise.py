# ABOUTME: Utility for adding norm-preserving Gaussian noise to latent vectors.
# ABOUTME: Used by experiments that perturb latent reasoning steps to study their sensitivity.

import torch


def add_norm_preserving_noise(
    tensor: torch.Tensor,
    noise_scale: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Add Gaussian noise scaled by noise_scale, then renormalize to preserve the original L2 norm.

    This changes the direction of the vector while keeping its magnitude constant.

    Args:
        tensor: Input tensor of shape (..., hidden_dim).
        noise_scale: Standard deviation multiplier for the Gaussian noise.
        generator: Optional torch.Generator for reproducible noise.

    Returns:
        Noisy tensor with the same L2 norm as the input.
    """
    original_norm = tensor.norm(dim=-1, keepdim=True)
    noise = torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        generator=generator,
    ) * noise_scale
    noisy = tensor + noise
    noisy_norm = noisy.norm(dim=-1, keepdim=True)
    # Avoid division by zero (shouldn't happen in practice)
    noisy_norm = noisy_norm.clamp(min=1e-8)
    noisy = noisy * (original_norm / noisy_norm)
    return noisy
