"""User-editable RMSnorm implementation for the baymax-zimage node.

Edit apply_rmsnorm and then run the baymax-zimage node with reload enabled.
The node patches ComfyUI RMSNorm.forward used by z-Image NextDiT, and adapts
that call into this apply_rmsnorm interface.
"""

import torch

@torch.compile(mode="max-autotune")
def apply_rmsnorm(x, weight=None, original_apply_rmsnorm=None, eps=1e-6):
    """
    Root Mean Square Normalization (RMSnorm) implementation.
    
    RMSnorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        x: Input tensor to normalize
        weight: Optional scaling factor. If None, uses unit scaling
        original_apply_rmsnorm: Original implementation fallback
        eps: Small epsilon value for numerical stability (default: 1e-6)
    
    Returns:
        Normalized tensor with same shape as input
    """
    #print(f'baymax run in user RMSNorm!')
    if x is None:
        if original_apply_rmsnorm is None:
            raise RuntimeError("x is None and no original_apply_rmsnorm fallback is available")
        return original_apply_rmsnorm(x, weight)

    if eps is None:
        eps = 1e-6
    
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