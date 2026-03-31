# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.utils import opcheck

DTYPES = [torch.half, torch.bfloat16, torch.float]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

# Mini test params for faster iteration
MINI_PYTEST_PARAMS = {
    "default": {
        "num_seqs": [2, 4],
        "vocab_size": [128, 512],
    },
}


def reference_apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of apply_repetition_penalties in Python."""
    result = logits.clone()
    num_seqs = logits.size(0)
    vocab_size = logits.size(1)

    for seq_idx in range(num_seqs):
        penalty = repetition_penalties[seq_idx].item()
        for vocab_idx in range(vocab_size):
            idx = seq_idx * vocab_size + vocab_idx
            if prompt_mask[idx].item() or output_mask[idx].item():
                logit = result[seq_idx, vocab_idx].item()
                if logit > 0:
                    result[seq_idx, vocab_idx] = logit / penalty
                else:
                    result[seq_idx, vocab_idx] = logit * penalty
    return result


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@pytest.mark.parametrize("num_seqs", [2, 8])
@pytest.mark.parametrize("vocab_size", [128, 1024, 50000])
@torch.inference_mode()
def test_apply_repetition_penalties(
    dtype: torch.dtype,
    device: str,
    num_seqs: int,
    vocab_size: int,
) -> None:
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    # Create test inputs
    logits = torch.randn(num_seqs, vocab_size, dtype=dtype)
    # Create masks with some repeated tokens
    prompt_mask = torch.zeros(num_seqs * vocab_size, dtype=torch.bool)
    output_mask = torch.zeros(num_seqs * vocab_size, dtype=torch.bool)

    # Set some tokens as repeated (prompt)
    for seq_idx in range(num_seqs):
        for vocab_idx in range(0, vocab_size, 10):
            idx = seq_idx * vocab_size + vocab_idx
            prompt_mask[idx] = True

    # Set some tokens as repeated (output)
    for seq_idx in range(num_seqs):
        for vocab_idx in range(5, vocab_size, 20):
            idx = seq_idx * vocab_size + vocab_idx
            output_mask[idx] = True

    # Random repetition penalties per sequence
    repetition_penalties = torch.ones(num_seqs, dtype=dtype) * 1.5

    # Make a copy for reference computation
    logits_ref = logits.clone()

    # Call the kernel
    torch.ops._C.apply_repetition_penalties_(logits, prompt_mask, output_mask,
                                             repetition_penalties)

    # Compute expected result
    expected = reference_apply_repetition_penalties(logits_ref, prompt_mask,
                                                    output_mask,
                                                    repetition_penalties)

    # Verify results
    torch.testing.assert_close(logits, expected, atol=1e-3, rtol=1e-3)

    # Op check for correctness validation
    opcheck(
        torch.ops._C.apply_repetition_penalties_,
        (logits, prompt_mask, output_mask, repetition_penalties),
    )


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_apply_repetition_penalties_edge_cases(
    dtype: torch.dtype,
    device: str,
) -> None:
    """Test edge cases like zero logits, negative logits, etc."""
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    num_seqs = 2
    vocab_size = 64

    # Test with zero logits
    logits = torch.zeros(num_seqs, vocab_size, dtype=dtype)
    prompt_mask = torch.zeros(num_seqs * vocab_size, dtype=torch.bool)
    output_mask = torch.zeros(num_seqs * vocab_size, dtype=torch.bool)
    repetition_penalties = torch.full((num_seqs, ), 1.5, dtype=dtype)

    # Mark some tokens as repeated
    prompt_mask[0] = True
    prompt_mask[vocab_size] = True

    torch.ops._C.apply_repetition_penalties_(logits, prompt_mask, output_mask,
                                             repetition_penalties)

    # Zero logits should stay zero (0 * penalty = 0)
    assert torch.all(logits == 0)

    # Test with negative logits
    logits_neg = torch.full((num_seqs, vocab_size), -1.0, dtype=dtype)
    logits_neg_copy = logits_neg.clone()

    torch.ops._C.apply_repetition_penalties_(logits_neg, prompt_mask,
                                             output_mask, repetition_penalties)

    # Negative logits should be multiplied by penalty
    expected = reference_apply_repetition_penalties(logits_neg_copy,
                                                    prompt_mask, output_mask,
                                                    repetition_penalties)
    torch.testing.assert_close(logits_neg, expected, atol=1e-3, rtol=1e-3)

    # Test with positive logits
    logits_pos = torch.full((num_seqs, vocab_size), 1.0, dtype=dtype)
    logits_pos_copy = logits_pos.clone()

    torch.ops._C.apply_repetition_penalties_(logits_pos, prompt_mask,
                                             output_mask, repetition_penalties)

    # Positive logits should be divided by penalty
    expected = reference_apply_repetition_penalties(logits_pos_copy,
                                                    prompt_mask, output_mask,
                                                    repetition_penalties)
    torch.testing.assert_close(logits_pos, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_apply_repetition_penalties_empty(
    dtype: torch.dtype,
    device: str,
) -> None:
    """Test with empty batch (num_seqs = 0)."""
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    num_seqs = 0
    vocab_size = 64

    logits = torch.randn(num_seqs, vocab_size, dtype=dtype)
    prompt_mask = torch.zeros(num_seqs * vocab_size, dtype=torch.bool)
    output_mask = torch.zeros(num_seqs * vocab_size, dtype=torch.bool)
    repetition_penalties = torch.ones(num_seqs, dtype=dtype)

    # Should not crash with empty batch
    torch.ops._C.apply_repetition_penalties_(logits, prompt_mask, output_mask,
                                             repetition_penalties)

    # Logits should be unchanged
    assert logits.numel() == 0
