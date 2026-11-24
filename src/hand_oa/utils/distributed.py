"""Lightweight PyTorch DistributedDataParallel (DDP) helpers shared across tracks."""

import os

import torch
import torch.distributed as dist


def init_distributed(device: torch.device | None = None) -> tuple[bool, int, int, int, torch.device]:
    """Initialize DDP from torchrun env vars when WORLD_SIZE > 1, else single-device mode."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and dist.is_available():
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            resolved_device = torch.device(f"cuda:{local_rank}")
        else:
            backend = "gloo"
            resolved_device = torch.device("cpu")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return True, rank, local_rank, world_size, resolved_device

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 0, 1, resolved_device


def cleanup_distributed() -> None:
    """Cleanly tear down DDP process group if active."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """Return True only on rank 0."""
    return rank == 0


def barrier_if_distributed(is_distributed: bool) -> None:
    """Synchronize all workers when running under DDP."""
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean_dict(
    metrics: dict[str, float],
    is_distributed: bool,
    world_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Average scalar metrics across all DDP workers."""
    if not is_distributed or world_size <= 1 or not metrics:
        return metrics
    keys = sorted(metrics.keys())
    tensor = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(world_size)
    return {k: float(tensor[i].item()) for i, k in enumerate(keys)}


def broadcast_bool(value: bool, is_distributed: bool, device: torch.device) -> bool:
    """Broadcast a boolean flag (e.g. early-stopping signal) from rank 0 to all ranks."""
    if not is_distributed or not dist.is_available() or not dist.is_initialized():
        return value
    flag = torch.tensor([1 if value else 0], dtype=torch.int64, device=device)
    dist.broadcast(flag, src=0)
    return bool(int(flag.item()) == 1)
