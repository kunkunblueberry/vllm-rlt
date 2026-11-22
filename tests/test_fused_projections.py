# SPDX-License-Identifier: Apache-2.0
"""Fusing projections that share an input row must not change what they compute.

The fused weight is a concatenation of the existing parameters, so an exact comparison
against the unfused call is the right gate: any real bug here shows up as a nonzero
difference, not as a rounding-level one. A tolerance would hide a mis-ordered
concatenation, which is the most likely failure mode.
"""

import pytest
import torch

from vllm_rlt.layers.fused_projections import FusedInputProjections
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM, OuroMLP

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


class _Triple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 32, bias=False)
        self.b = torch.nn.Linear(64, 16, bias=False)
        self.c = torch.nn.Linear(64, 16, bias=False)
        self.fused = FusedInputProjections(("a", "b", "c"), self)


def test_fused_weight_is_the_concatenation_in_source_order():
    layer = _Triple()
    fused = layer.fused.weight()
    assert fused.shape == (64, 64)
    assert torch.equal(fused[:32], layer.a.weight)
    assert torch.equal(fused[32:48], layer.b.weight)
    assert torch.equal(fused[48:], layer.c.weight)


def test_split_matches_separate_projections_exactly():
    torch.manual_seed(0)
    layer = _Triple()
    hidden = torch.randn(3, 64)
    expected = (layer.a(hidden), layer.b(hidden), layer.c(hidden))
    fused = layer.fused(hidden)
    got = (fused[:, :32], fused[:, 32:48], fused[:, 48:])
    for g, e in zip(got, expected):
        assert torch.equal(g, e)


def test_fused_weight_rebuilds_after_parameter_replacement():
    """Checkpoint reload and dtype conversion replace parameters; a stale cache would
    silently keep computing with the old weights."""
    layer = _Triple()
    first = layer.fused.weight()
    with torch.no_grad():
        layer.a.weight = torch.nn.Parameter(torch.zeros_like(layer.a.weight), requires_grad=False)
    second = layer.fused.weight()
    assert second.data_ptr() != first.data_ptr()
    assert torch.equal(second[:32], layer.a.weight)
    layer.half()
    third = layer.fused.weight()
    assert third.dtype == torch.float16
    # _apply converts in place, so the signature only changes via dtype here.
    assert torch.equal(third[32:48], layer.b.weight)

def test_state_dict_is_unchanged_by_fusion():
    """The fused matrices are derived, so they must not appear as checkpoint keys."""
    config = OuroConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, head_dim=8,
        max_position_embeddings=128, total_ut_steps=2, bos_token_id=0, eos_token_id=0,
    )
    model = OuroForCausalLM(config)
    keys = set(model.state_dict())
    assert not any("qkv_proj" in k or "gate_up_proj" in k for k in keys), sorted(keys)
    expected = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    }
    assert expected <= keys


@requires_cuda
def test_mlp_output_is_bitwise_identical_to_the_unfused_expression():
    torch.manual_seed(1)
    config = OuroConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16,
        max_position_embeddings=128, total_ut_steps=2, bos_token_id=0, eos_token_id=0,
    )
    mlp = OuroMLP(config).to("cuda", torch.bfloat16).eval()
    hidden = torch.randn(4, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = mlp.down_proj(
            torch.nn.functional.silu(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
        )
        got = mlp(hidden)
    assert torch.equal(got, expected), (
        f"max diff {(got.float() - expected.float()).abs().max().item()}"
    )
