"""Shared GPU memory helpers for sequential model loading."""

import gc


def empty_cuda_cache():
    """Force Python GC and release cached CUDA allocations."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def cuda_free_gb() -> float:
    """Return free GPU memory in GB, or 0 if CUDA unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return free / (1024 ** 3)
    except Exception:
        pass
    return 0.0
