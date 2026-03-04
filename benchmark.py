#!/usr/bin/env python3
"""
End-to-end benchmarking script for CS336 Assignment 2 §1.1.3.

Times forward (and optionally backward) passes of the basics Transformer
with configurable model size, context length, warmup steps, and number of
measurement steps. Uses timeit.default_timer() and torch.cuda.synchronize()
for accurate GPU timing.

Example:
  uv run python benchmark.py --size small --context_length 128 --warmup 5 --steps 10
  uv run python benchmark.py --size medium --context_length 256 --forward_only
  uv run nsys profile -o result python benchmark.py --size small --context_length 512
"""

from __future__ import annotations

import argparse
import timeit
import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F

import cs336_basics.model as basics_model
from cs336_basics.optimizer import AdamW
from cs336_systems.config import (
    BATCH_SIZE,
    CONTEXT_LENGTHS,
    MODEL_SIZES,
    VOCAB_SIZE,
    get_model_kwargs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Transformer forward/backward passes.")
    p.add_argument(
        "--size",
        type=str,
        choices=list(MODEL_SIZES),
        default="small",
        help="Model size from Table 1 (default: small)",
    )
    p.add_argument(
        "--context_length",
        type=int,
        default=128,
        help="Sequence length (default: 128)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    p.add_argument(
        "--vocab_size",
        type=int,
        default=VOCAB_SIZE,
        help=f"Vocabulary size (default: {VOCAB_SIZE})",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        metavar="w",
        help="Number of warm-up steps before timing (default: 5)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=10,
        metavar="n",
        help="Number of steps to time (default: 10)",
    )
    p.add_argument(
        "--forward_only",
        action="store_true",
        help="Time only forward pass; if not set, time forward + backward",
    )
    p.add_argument(
        "--training",
        action="store_true",
        help="Include AdamW optimizer step (full training step)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args()


def get_random_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random input IDs and targets for benchmarking (no real data)."""
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)
    # Targets: shift by 1 for a valid cross-entropy target (arbitrary for timing)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)
    return x, y


def run_one_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    forward_only: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Run one forward pass, and backward + optimizer step if not forward_only."""
    with nvtx.range("forward"):
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-1,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if not forward_only:
        with nvtx.range("backward"):
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    if optimizer is not None:
        with nvtx.range("optimizer_step"):
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()


def time_steps(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    forward_only: bool,
    warmup: int,
    steps: int,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
) -> list[float]:
    """Run warmup steps, then time `steps` steps. Returns list of timings in seconds."""
    model.train()
    timings: list[float] = []
    is_cuda = isinstance(device, torch.device) and device.type == "cuda" or device == "cuda"

    # Warm-up
    for i in range(warmup):
        with nvtx.range(f"warmup_{i}"):
            run_one_step(model, x, y, forward_only, optimizer)
            if not forward_only:
                model.zero_grad(set_to_none=True)
            if is_cuda:
                torch.cuda.synchronize()

    # Timed steps
    for i in range(steps):
        if not forward_only:
            model.zero_grad(set_to_none=True)
        if is_cuda:
            torch.cuda.synchronize()
        start = timeit.default_timer()
        with nvtx.range(f"step_{i}"):
            run_one_step(model, x, y, forward_only, optimizer)
        if is_cuda:
            torch.cuda.synchronize()
        end = timeit.default_timer()
        timings.append(end - start)

    return timings


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Build model from Table 1 config
    kwargs = get_model_kwargs(
        args.size, args.context_length, vocab_size=args.vocab_size
    )
    model = basics_model.BasicsTransformerLM(**kwargs)
    model = model.to(device)

    optimizer = None
    if args.training:
        optimizer = AdamW(model.parameters(), lr=1e-4)

    x, y = get_random_batch(
        args.batch_size,
        args.context_length,
        args.vocab_size,
        device,
    )

    mode = "forward only" if args.forward_only else ("training (fwd+bwd+adamw)" if args.training else "forward + backward")
    print(f"Model: {args.size}, context_length={args.context_length}, batch_size={args.batch_size}")
    print(f"Warmup: {args.warmup} steps, timing: {args.steps} steps ({mode})")
    print(f"Device: {device}")

    timings = time_steps(
        model, x, y,
        forward_only=args.forward_only,
        warmup=args.warmup,
        steps=args.steps,
        device=device,
        optimizer=optimizer,
    )

    mean_s = sum(timings) / len(timings)
    variance = sum((t - mean_s) ** 2 for t in timings) / len(timings)
    std_s = variance ** 0.5
    mean_ms = mean_s * 1000
    std_ms = std_s * 1000

    print(f"\nTiming: mean = {mean_ms:.2f} ms, std = {std_ms:.2f} ms")
    print(f"        mean = {mean_s:.4f} s,  std = {std_s:.4f} s")


if __name__ == "__main__":
    main()
