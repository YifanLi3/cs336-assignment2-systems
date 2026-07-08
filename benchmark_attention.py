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

Modes:
  --mode eager    : uncompiled reference
  --mode compile  : torch.compile(scaled_dot_product_attention)
  --mode both     : run both back-to-back and print a comparison table (default)

Usage:
    uv run python benchmark_attention.py
    uv run python benchmark_attention.py --mode compile
    uv run python benchmark_attention.py --mode both --csv results_attention.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import timeit
from dataclasses import dataclass
from typing import Callable, Optional

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
    attn_fn: Callable,
) -> Result:
    """Benchmark one (d_model, seq_len) configuration using `attn_fn(Q,K,V)`."""
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
        for _ in range(warmup):
            out = attn_fn(Q, K, V)
            torch.cuda.synchronize()
            del out

        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        for _ in range(steps):
            out = attn_fn(Q, K, V)
            torch.cuda.synchronize()
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
        for _ in range(warmup):
            out = attn_fn(Q, K, V)
            loss = out.sum()
            loss.backward()
            Q.grad = K.grad = V.grad = None
            torch.cuda.synchronize()

        # ---- Memory in use just before backward starts ----
        free_cuda()
        torch.cuda.reset_peak_memory_stats()
        out = attn_fn(Q, K, V)
        loss = out.sum()
        torch.cuda.synchronize()
        res.mem_before_bw_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        # First backward (uses the graph we just built).
        t0 = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        total = timeit.default_timer() - t0
        Q.grad = K.grad = V.grad = None

        # Remaining (steps - 1) backwards: each needs a fresh forward.
        for _ in range(steps - 1):
            out = attn_fn(Q, K, V)
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
    p.add_argument("--mode", choices=["eager", "compile", "both"], default="both")
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


def _fmt(x: Optional[float], w: int = 8, prec: int = 3) -> str:
    return f"{x:>{w}.{prec}f}" if x is not None else f"{'-':>{w}}"


def run_sweep(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    attn_fn: Callable,
    label: str,
) -> list[Result]:
    print(f"\n### Mode: {label} ###")
    results: list[Result] = []
    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            print(f"  d_model={d_model:4d}, seq_len={seq_len:5d} ... ", end="", flush=True)
            r = bench_config(
                d_model=d_model,
                seq_len=seq_len,
                batch_size=args.batch_size,
                warmup=args.warmup,
                steps=args.steps,
                device=device,
                dtype=dtype,
                attn_fn=attn_fn,
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
    return results


def print_single_table(results: list[Result], label: str, dtype_str: str) -> None:
    print("\n" + "=" * 90)
    print(f"Summary ({label}, batch=8, single-head, dtype={dtype_str})")
    print("=" * 90)
    header = (
        f"{'d_model':>8} {'seq_len':>8} {'fw (ms)':>10} {'bw (ms)':>10} "
        f"{'mem_before_bw (MB)':>22} {'status':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        fw = f"{r.fw_ms:.3f}" if r.fw_ms is not None else "-"
        bw = f"{r.bw_ms:.3f}" if r.bw_ms is not None else "-"
        mem = f"{r.mem_before_bw_mb:.2f}" if r.mem_before_bw_mb is not None else "-"
        print(f"{r.d_model:>8} {r.seq_len:>8} {fw:>10} {bw:>10} {mem:>22} {r.status:>14}")


def print_comparison_table(
    eager: list[Result], compiled: list[Result], dtype_str: str
) -> None:
    print("\n" + "=" * 118)
    print(f"Eager vs. torch.compile  (batch=8, single-head, dtype={dtype_str})")
    print("=" * 118)
    header = (
        f"{'d_model':>7} {'seq_len':>7} "
        f"{'fw eager':>10} {'fw comp':>10} {'fw speedup':>11} "
        f"{'bw eager':>10} {'bw comp':>10} {'bw speedup':>11} "
        f"{'status':>18}"
    )
    print(header)
    print("-" * len(header))
    for e, c in zip(eager, compiled):
        assert e.d_model == c.d_model and e.seq_len == c.seq_len

        def spd(a: Optional[float], b: Optional[float]) -> str:
            if a is None or b is None or b == 0:
                return "-"
            return f"{a / b:.2f}x"

        st = (
            "ok" if e.status == "ok" and c.status == "ok"
            else f"e={e.status}|c={c.status}"
        )
        print(
            f"{e.d_model:>7} {e.seq_len:>7} "
            f"{_fmt(e.fw_ms, 10):>10} {_fmt(c.fw_ms, 10):>10} {spd(e.fw_ms, c.fw_ms):>11} "
            f"{_fmt(e.bw_ms, 10):>10} {_fmt(c.bw_ms, 10):>10} {spd(e.bw_ms, c.bw_ms):>11} "
            f"{st:>18}"
        )


def write_csv(path: str, rows: list[tuple]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"\nWrote results to {path}")


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required to run this benchmark."
    device = torch.device("cuda")
    dtype = dtype_from_str(args.dtype)

    dev_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Device: {dev_name}  ({total_mem_gb:.1f} GB)")
    print(
        f"mode={args.mode}, batch_size={args.batch_size}, warmup={args.warmup}, "
        f"steps={args.steps}, dtype={args.dtype}"
    )

    eager_fn = scaled_dot_product_attention
    compiled_fn = torch.compile(scaled_dot_product_attention) if args.mode != "eager" else None

    eager_results: list[Result] = []
    compiled_results: list[Result] = []

    if args.mode in ("eager", "both"):
        eager_results = run_sweep(args, device, dtype, eager_fn, label="eager")
    if args.mode in ("compile", "both"):
        # Fresh dynamo state so we don't inherit anything from an earlier run.
        import torch._dynamo as dynamo
        dynamo.reset()
        compiled_results = run_sweep(args, device, dtype, compiled_fn, label="torch.compile")

    # -------------- Pretty-print --------------
    if args.mode == "eager":
        print_single_table(eager_results, "eager", args.dtype)
    elif args.mode == "compile":
        print_single_table(compiled_results, "torch.compile", args.dtype)
    else:  # both
        print_single_table(eager_results, "eager", args.dtype)
        print_single_table(compiled_results, "torch.compile", args.dtype)
        print_comparison_table(eager_results, compiled_results, args.dtype)

    # -------------- Optional CSV --------------
    if args.csv:
        rows: list[tuple] = [("mode", "d_model", "seq_len", "fw_ms", "bw_ms", "mem_before_bw_mb", "status")]
        for r in eager_results:
            rows.append(("eager", r.d_model, r.seq_len, r.fw_ms, r.bw_ms, r.mem_before_bw_mb, r.status))
        for r in compiled_results:
            rows.append(("compile", r.d_model, r.seq_len, r.fw_ms, r.bw_ms, r.mem_before_bw_mb, r.status))
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
