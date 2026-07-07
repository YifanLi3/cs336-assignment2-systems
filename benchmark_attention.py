#!/usr/bin/env python3
"""
Benchmark attention forward/backward passes across (d_model, seq_len) configurations.

Spec (CS336 Assignment 2):
  (a) Fix batch size to 8, no multi-head (single-head, i.e. shape [B, L, d]).
  (b) Cartesian product of d_model in [16, 32, 64, 128] and
      seq_len in [256, 1024, 4096, 8192, 16384].
  (c) Random Q, K, V.
  (d) Time 100 forward passes.
  (e) Measure memory in use before backward starts; time 100 backward passes.
  (f) Warm up; torch.cuda.synchronize() after each pass.

OOM configurations are caught, reported, and the run continues.

Usage:
    uv run python benchmark_attention.py
    uv run python benchmark_attention.py --steps 100 --warmup 10 --dtype float32
    uv run python benchmark_attention.py --csv results_attention.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import timeit
from dataclasses import dataclass
from typing import Optional

import torch

from cs336_basics.model import scaled_dot_product_attention

BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


@dataclass
class Result:
    d_model: int
    seq_len: int
    fw_ms: Optional[float] = None
    bw_ms: Optional[float] = None
    mem_before_bw_mb: Optional[float] = None
    status: str = "ok"  # "ok", "OOM (fwd)", "OOM (bwd)", "OOM (alloc)"


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def bench_config(
    d_model: int,
    seq_len: int,
    batch_size: int,
    warmup: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Result:
    """Benchmark one (d_model, seq_len) configuration."""
    res = Result(d_model=d_model, seq_len=seq_len)

    # --- Allocate Q, K, V ---
    try:
        Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    except torch.cuda.OutOfMemoryError:
        res.status = "OOM (alloc)"
        free_cuda()
        return res

    # ==================== FORWARD ====================
    try:
        # Warmup (with grad tracking so it matches the real forward path)
        for _ in range(warmup):
            out = scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()
            del out

        # Timed forward passes
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        for _ in range(steps):
            out = scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()
            # Note: `out` and its autograd graph get overwritten each iter,
            # so memory does not grow across iterations.
        t1 = timeit.default_timer()
        res.fw_ms = (t1 - t0) / steps * 1000.0
        del out
    except torch.cuda.OutOfMemoryError:
        res.status = "OOM (fwd)"
        del Q, K, V
        free_cuda()
        return res

    # ==================== BACKWARD ====================
    try:
        # Warmup backward
        for _ in range(warmup):
            out = scaled_dot_product_attention(Q, K, V)
            loss = out.sum()
            loss.backward()
            Q.grad = K.grad = V.grad = None
            torch.cuda.synchronize()

        # ---- Measure memory in use just before backward starts ----
        free_cuda()
        torch.cuda.reset_peak_memory_stats()
        out = scaled_dot_product_attention(Q, K, V)
        loss = out.sum()
        torch.cuda.synchronize()
        res.mem_before_bw_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        # First backward (uses the graph we just built).
        t0 = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        total = timeit.default_timer() - t0
        Q.grad = K.grad = V.grad = None

        # Remaining (steps - 1) backward passes: each needs a fresh forward.
        for _ in range(steps - 1):
            out = scaled_dot_product_attention(Q, K, V)
            loss = out.sum()
            torch.cuda.synchronize()
            t0 = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            total += timeit.default_timer() - t0
            Q.grad = K.grad = V.grad = None

        res.bw_ms = total / steps * 1000.0
    except torch.cuda.OutOfMemoryError:
        res.status = "OOM (bwd)"

    del Q, K, V
    free_cuda()
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark scaled-dot-product attention.")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    p.add_argument("--d_models", type=int, nargs="+", default=D_MODELS)
    p.add_argument("--seq_lens", type=int, nargs="+", default=SEQ_LENS)
    return p.parse_args()


def dtype_from_str(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[s]


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required to run this benchmark."
    device = torch.device("cuda")
    dtype = dtype_from_str(args.dtype)

    dev_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Device: {dev_name}  ({total_mem_gb:.1f} GB)")
    print(
        f"batch_size={args.batch_size}, warmup={args.warmup}, steps={args.steps}, dtype={args.dtype}\n"
    )

    results: list[Result] = []
    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            print(f"Running d_model={d_model:4d}, seq_len={seq_len:5d} ... ", end="", flush=True)
            r = bench_config(
                d_model=d_model,
                seq_len=seq_len,
                batch_size=args.batch_size,
                warmup=args.warmup,
                steps=args.steps,
                device=device,
                dtype=dtype,
            )
            results.append(r)
            if r.status == "ok":
                print(
                    f"fw={r.fw_ms:7.3f} ms | "
                    f"mem_before_bw={r.mem_before_bw_mb:8.2f} MB | "
                    f"bw={r.bw_ms:7.3f} ms"
                )
            else:
                print(f"{r.status}")

    # -------------- Pretty-print summary table --------------
    print("\n" + "=" * 90)
    print("Summary (attention, batch=8, single-head, dtype=%s)" % args.dtype)
    print("=" * 90)
    header = f"{'d_model':>8} {'seq_len':>8} {'fw (ms)':>10} {'bw (ms)':>10} {'mem_before_bw (MB)':>22} {'status':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        fw = f"{r.fw_ms:.3f}" if r.fw_ms is not None else "-"
        bw = f"{r.bw_ms:.3f}" if r.bw_ms is not None else "-"
        mem = f"{r.mem_before_bw_mb:.2f}" if r.mem_before_bw_mb is not None else "-"
        print(f"{r.d_model:>8} {r.seq_len:>8} {fw:>10} {bw:>10} {mem:>22} {r.status:>14}")

    # -------------- Optional CSV output --------------
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["d_model", "seq_len", "fw_ms", "bw_ms", "mem_before_bw_mb", "status"])
            for r in results:
                w.writerow([r.d_model, r.seq_len, r.fw_ms, r.bw_ms, r.mem_before_bw_mb, r.status])
        print(f"\nWrote results to {args.csv}")


if __name__ == "__main__":
    main()
