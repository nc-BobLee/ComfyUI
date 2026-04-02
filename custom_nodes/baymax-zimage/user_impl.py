"""User-editable RMSnorm implementation for the baymax-zimage node.

Edit apply_rmsnorm and then run the baymax-zimage node with reload enabled.
The node patches ComfyUI RMSNorm.forward used by z-Image NextDiT, and adapts
that call into this apply_rmsnorm interface.
"""

import torch

@torch.compile(mode="max-autotune")
def apply_rmsnorm(x, weight=None, eps=1e-6):
    """
    Root Mean Square Normalization implementation.

    RMSnorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    # Store original dtype
    original_dtype = x.dtype
    
    # Ensure x is float32 for numerical stability
    x_float = x.to(dtype=torch.float32)
    
    # Calculate RMS along the last dimension
    # RMS(x) = sqrt(mean(x^2) + eps)
    rms = torch.sqrt((x_float ** 2).mean(dim=-1, keepdim=True) + eps)
    
    # Normalize by RMS
    x_norm = x_float / rms
    
    # Apply scaling factor (weight) if provided
    if weight is not None:
        weight_float = weight.to(dtype=torch.float32) if weight.dtype != torch.float32 else weight
        x_norm = x_norm * weight_float
    
    # Convert back to original dtype
    return x_norm.to(dtype=original_dtype)


@torch.compile(mode="max-autotune")
def apply_rope(xq, xk, freqs_cis):
    """Rotary positional embedding implementation compatible with flux/lumina."""

    def _apply_single(x):
        if x is None:
            return None

        x_work = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
        fc = freqs_cis

        if x_work.shape[2] != 1 and fc.shape[2] != 1 and x_work.shape[2] != fc.shape[2]:
            fc = fc[:, :, :x_work.shape[2]]

        x_out = fc[..., 0] * x_work[..., 0]
        x_out.addcmul_(fc[..., 1], x_work[..., 1])
        return x_out.reshape(*x.shape).type_as(x)

    return _apply_single(xq), _apply_single(xk)