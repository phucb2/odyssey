"""Unit tests for scratch Qwen3-style components in minigpt/qwen3_scatch.py."""

from __future__ import annotations

import pytest
import torch

from odyssey.experiments.minigpt.qwen3_scatch import (
    MLP,
    CharLMDataset,
    DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3LMConfig,
    Qwen3Model,
    Qwen3ScratchConfig,
    QwenAttention,
    RotaryEmbedding,
    Tokenizer,
    apply_rope,
    build_tokenizer_and_data,
    causal_lm_loss,
    create_qwen3_dls,
    get_batch,
    repeat_kv,
    rotate_half,
)


def _tiny_cfg(**overrides) -> Qwen3ScratchConfig:
    base = dict(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    base.update(overrides)
    return Qwen3ScratchConfig(**base)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_fit_encode_decode_roundtrip():
    tok = Tokenizer()
    tok.fit(["hi", "hey"])
    ids = tok.encode("hi")
    assert ids[0].item() == tok.bos
    assert ids[-1].item() == tok.eos
    assert tok.decode(ids) == "hi"
    assert tok.vocab_size >= 5  # pad/bos/eos + h,i,e,y


def test_tokenizer_unknown_char_raises():
    tok = Tokenizer()
    with pytest.raises(ValueError, match="unknown character"):
        tok.encode("z")


def test_tokenizer_skips_special_tokens_on_decode():
    tok = Tokenizer()
    tok.fit(["a"])
    ids = torch.tensor([tok.bos, tok.stoi["a"], tok.pad, tok.eos])
    assert tok.decode(ids) == "a"


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def test_rotate_half_swaps_negates():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.allclose(out, expected)


def test_rotary_embedding_shapes():
    rope = RotaryEmbedding(base=10000.0, dim=8)
    pos = torch.arange(5).unsqueeze(0).expand(2, -1)
    cos, sin = rope(pos)
    assert cos.shape == (2, 5, 8)
    assert sin.shape == (2, 5, 8)
    assert torch.allclose(cos.pow(2) + sin.pow(2), torch.ones_like(cos), atol=1e-5)


def test_apply_rope_preserves_norm():
    """RoPE is an orthogonal transform — vector L2 norms stay the same."""
    bs, n_heads, seq, dim = 2, 3, 4, 8
    q = torch.randn(bs, n_heads, seq, dim)
    k = torch.randn(bs, n_heads, seq, dim)
    rope = RotaryEmbedding(10000.0, dim)
    cos, sin = rope(torch.arange(seq).unsqueeze(0).expand(bs, -1))
    q2, k2 = apply_rope(q, k, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k2.norm(dim=-1), atol=1e-5)


def test_apply_rope_position_zero_is_identity_rotation():
    """At position 0, cos=1 and sin=0 so RoPE is identity."""
    q = torch.randn(1, 2, 1, 4)
    k = torch.randn(1, 2, 1, 4)
    rope = RotaryEmbedding(10000.0, 4)
    cos, sin = rope(torch.zeros(1, 1, dtype=torch.long))
    q2, k2 = apply_rope(q, k, cos, sin)
    assert torch.allclose(q, q2, atol=1e-5)
    assert torch.allclose(k, k2, atol=1e-5)


# ---------------------------------------------------------------------------
# repeat_kv / GQA
# ---------------------------------------------------------------------------


def test_repeat_kv_expands_head_dim():
    x = torch.arange(8).view(1, 2, 1, 4).float()  # (bs, n_kv, seq, dim)
    out = repeat_kv(x, 2)
    assert out.shape == (1, 4, 1, 4)
    # Head 0 and 1 are copies of kv-head 0; 2 and 3 copies of kv-head 1.
    assert torch.equal(out[:, 0], out[:, 1])
    assert torch.equal(out[:, 2], out[:, 3])
    assert not torch.equal(out[:, 0], out[:, 2])


def test_repeat_kv_n_rep_one_is_noop():
    x = torch.randn(2, 3, 5, 4)
    assert repeat_kv(x, 1) is x


# ---------------------------------------------------------------------------
# Attention / MLP / Model
# ---------------------------------------------------------------------------


def test_attention_gqa_shapes_and_forward():
    attn = QwenAttention(hidden_size=32, num_heads=4, num_kv_heads=2, head_dim=8)
    assert attn.q_proj.out_features == 4 * 8
    assert attn.k_proj.out_features == 2 * 8
    assert attn.v_proj.out_features == 2 * 8

    bs, seq = 2, 5
    x = torch.randn(bs, seq, 32)
    rope = RotaryEmbedding(10000.0, 8)
    cos, sin = rope(torch.arange(seq).unsqueeze(0).expand(bs, -1))
    out = attn(x, cos, sin)
    assert out.shape == (bs, seq, 32)
    assert torch.isfinite(out).all()


def test_attention_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        QwenAttention(32, num_heads=6, num_kv_heads=4, head_dim=8)


def test_attention_causal_mask_blocks_future():
    torch.manual_seed(0)
    attn = QwenAttention(hidden_size=16, num_heads=2, num_kv_heads=2, head_dim=8, causal=True)
    # Freeze projections so we can inspect scores via a hook-free path:
    # compare outputs when only the last token of input changes.
    bs, seq = 1, 4
    x = torch.randn(bs, seq, 16)
    rope = RotaryEmbedding(10000.0, 8)
    cos, sin = rope(torch.arange(seq).unsqueeze(0))

    out1 = attn(x, cos, sin)
    x2 = x.clone()
    x2[:, -1] = torch.randn(16)
    out2 = attn(x2, cos, sin)
    # Positions before the last token must be unchanged under causal attention.
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
    assert not torch.allclose(out1[:, -1], out2[:, -1], atol=1e-5)


def test_attention_padding_mask():
    attn = QwenAttention(hidden_size=16, num_heads=2, num_kv_heads=1, head_dim=8)
    bs, seq = 1, 3
    x = torch.randn(bs, seq, 16)
    rope = RotaryEmbedding(10000.0, 8)
    cos, sin = rope(torch.arange(seq).unsqueeze(0))
    mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)
    out = attn(x, cos, sin, attention_mask=mask)
    assert out.shape == (bs, seq, 16)
    assert torch.isfinite(out).all()


def test_mlp_shape():
    mlp = MLP(32, 64)
    x = torch.randn(2, 5, 32)
    assert mlp(x).shape == x.shape


def test_decoder_layer_and_model_forward():
    cfg = _tiny_cfg()
    model = Qwen3Model(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 7))
    with torch.no_grad():
        out = model(ids)
    assert out.shape == (2, 7, cfg.hidden_size)
    assert torch.isfinite(out).all()


def test_model_uses_rope_theta_not_max_positions():
    cfg = _tiny_cfg(rope_theta=1234.0, max_position_embeddings=999)
    model = Qwen3Model(cfg)
    assert model.rope.base == 1234.0


def test_model_with_attention_mask():
    cfg = _tiny_cfg(num_hidden_layers=1)
    model = Qwen3Model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
    out = model(ids, attention_mask=mask)
    assert out.shape == (2, 5, cfg.hidden_size)


def test_decoder_layer_residual_shapes():
    cfg = _tiny_cfg(num_hidden_layers=1)
    layer = DecoderLayer(cfg)
    x = torch.randn(2, 6, cfg.hidden_size)
    rope = RotaryEmbedding(cfg.rope_theta, cfg.head_dim)
    cos, sin = rope(torch.arange(6).unsqueeze(0).expand(2, -1))
    out = layer(x, cos, sin)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Dataset / LM training helpers
# ---------------------------------------------------------------------------


def test_encode_chars_no_specials():
    tok = Tokenizer()
    tok.fit("abc")
    ids = tok.encode_chars("ab")
    assert ids.tolist() == [tok.stoi["a"], tok.stoi["b"]]
    assert tok.bos not in ids.tolist()


def test_char_lm_dataset_and_get_batch():
    data = torch.arange(20)
    ds = CharLMDataset(data, block_size=8)
    assert len(ds) == 12
    x, y = ds[0]
    assert x.tolist() == list(range(8))
    assert y.tolist() == list(range(1, 9))

    torch.manual_seed(0)
    xb, yb = get_batch(data, batch_size=4, block_size=8)
    assert xb.shape == (4, 8) and yb.shape == (4, 8)
    assert torch.equal(yb[:, :-1], xb[:, 1:])


def test_build_tokenizer_and_dls_from_text():
    text = "Trăm năm trong cõi người ta. " * 20
    cfg = Qwen3LMConfig()
    cfg.data.block_size = 16
    cfg.data.batch_size = 4
    cfg.data.valid_frac = 0.2
    tok, train_data, valid_data = build_tokenizer_and_data(cfg, text=text)
    assert tok.vocab_size > 3
    assert len(train_data) > cfg.data.block_size
    dls, tok2 = create_qwen3_dls(cfg, text=text)
    assert tok2.vocab_size == tok.vocab_size
    xb, yb = next(iter(dls.train))
    assert xb.shape == (4, 16) and yb.shape == (4, 16)


def test_causal_lm_forward_and_loss():
    cfg = _tiny_cfg(vocab_size=40, num_hidden_layers=1)
    model = Qwen3ForCausalLM(cfg)
    x = torch.randint(0, 40, (2, 7))
    y = torch.randint(0, 40, (2, 7))
    logits = model(x)
    assert logits.shape == (2, 7, 40)
    loss = causal_lm_loss(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
