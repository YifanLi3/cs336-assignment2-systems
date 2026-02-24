"""Model and benchmark configuration for CS336 Assignment 2 (§1.1.2 Table 1)."""

from __future__ import annotations

# Assignment-wide: vocab_size=10_000, batch_size=4, varying context lengths
VOCAB_SIZE = 10_000
BATCH_SIZE = 4
CONTEXT_LENGTHS = (128, 256, 512, 1024)
DEFAULT_ROPE_THETA = 10000.0

# Table 1: Specifications of different model sizes (Size -> d_model, d_ff, num_layers, num_heads)
MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def get_model_kwargs(
    size: str,
    context_length: int,
    vocab_size: int = VOCAB_SIZE,
    rope_theta: float = DEFAULT_ROPE_THETA,
) -> dict:
    """Keyword arguments for cs336_basics.model.BasicsTransformerLM."""
    if size not in MODEL_SIZES:
        raise KeyError(f"Unknown size {size!r}. Choose from {list(MODEL_SIZES)}")
    spec = MODEL_SIZES[size]
    return {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": spec["d_model"],
        "num_layers": spec["num_layers"],
        "num_heads": spec["num_heads"],
        "d_ff": spec["d_ff"],
        "rope_theta": rope_theta,
    }
