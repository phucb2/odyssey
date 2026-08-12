"""Tests for two-step math char LM training."""

from __future__ import annotations

from pathlib import Path

from types import SimpleNamespace

import torch

from odyssey.experiments.math_corpus import generate_dataset
from odyssey.experiments.minigpt.math_lm import (
    MathQADataset,
    build_qa_examples,
    collate_math_qa,
    equation_prompt_answer,
)
from odyssey.experiments.minigpt.qwen3_scatch import (
    Qwen3ForCausalLM,
    Qwen3LMConfig,
    Qwen3ScratchConfig,
    Tokenizer,
    load_qwen3_checkpoint,
    save_qwen3_checkpoint,
)


def test_equation_prompt_answer_canonical() -> None:
    row = {
        "op": "plus",
        "domain": "I",
        "lhs": [9, 1],
        "rhs": [16, 1],
        "result": [25, 1],
        "sentences": [],
    }
    prompt, answer = equation_prompt_answer(row)
    assert prompt == "nine plus sixteen equals"
    assert answer == "twenty five"


def test_math_qa_dataset_masks_prompt_tokens() -> None:
    tok = Tokenizer()
    tok.fit(["nine plus sixteen equals twenty five"])
    examples = [("nine plus sixteen equals", "twenty five")]
    item = MathQADataset(examples, tok)[0]
    input_ids, labels = item
    assert len(input_ids) == len(labels)
    assert (labels == -100).sum().item() == len("nine plus sixteen equals") - 1
    assert labels[-1].item() == tok.eos


def test_collate_math_qa_pads_batch() -> None:
    tok = Tokenizer()
    tok.fit(["abc def"])
    ds = MathQADataset([("ab", "c"), ("a", "bc")], tok)
    batch = collate_math_qa([ds[0], ds[1]], pad_id=tok.pad)
    input_ids, labels, attn = batch
    assert input_ids.shape[0] == 2
    assert labels.shape == input_ids.shape
    assert attn.shape == input_ids.shape


def test_build_qa_examples_from_generated_dataset(tmp_path: Path) -> None:
    root = generate_dataset(n_equations=5, max_int=5, max_den=4, seed=1, output=tmp_path)
    from odyssey.experiments.math_corpus import load_equations

    rows = load_equations(root)
    examples = build_qa_examples(rows)
    assert len(examples) == len(rows)
    for (prompt, answer), row in zip(examples, rows):
        assert prompt.endswith("equals")
        assert answer


def test_load_qwen3_checkpoint_roundtrip(tmp_path: Path) -> None:
    tok = Tokenizer()
    tok.fit("nine plus sixteen equals twenty five")
    cfg = Qwen3LMConfig()
    cfg.data.block_size = 16
    model_cfg = Qwen3ScratchConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    lm = Qwen3ForCausalLM(model_cfg)
    learn = SimpleNamespace(model=lm, tokenizer=tok)
    ckpt = save_qwen3_checkpoint(learn, cfg, path=tmp_path / "math.pt")
    loaded_model, loaded_tok, loaded_cfg, _ = load_qwen3_checkpoint(ckpt)
    assert loaded_tok.vocab_size == tok.vocab_size
    assert loaded_cfg.data.block_size == cfg.data.block_size
    x = torch.randint(0, tok.vocab_size, (1, 4))
    assert torch.allclose(lm(x), loaded_model(x), atol=1e-6)
