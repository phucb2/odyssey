"""Two-step math char LM: corpus pretrain then answer-only fine-tune.

Train via ``python -m odyssey.experiments.minigpt.math_lm --stage pretrain|tune|eval``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from odyssey.experiments.math_corpus import (
    Rational,
    default_dataset_dir,
    load_equations_split,
    value_to_text,
)
from odyssey.experiments.minigpt.qwen3_scatch import (
    Qwen3DataLoaders,
    Qwen3ForCausalLM,
    Qwen3LMConfig,
    Tokenizer,
    causal_lm_loss,
    compose_qwen3_scratch_config,
    create_qwen3_dls,
    load_qwen3_checkpoint,
    save_qwen3_checkpoint,
)
from odyssey.paths import DEFAULT_PROJECT, checkpoint_path, dataset_dir
from odyssey.training.callbacks import TrainCB, default_cbs
from odyssey.training.learner import Learner

OP_PHRASE = {
    "plus": "plus",
    "minus": "minus",
    "multiply": "times",
    "divide": "divided by",
}

IGNORE_INDEX = -100
PRETRAIN_CHECKPOINT = "qwen3_math_pretrain.pt"
TUNE_CHECKPOINT = "qwen3_math_tune.pt"


@dataclass
class MathTuneConfig:
    holdout_frac: float = 0.1
    max_answer_tokens: int = 64
    pretrain_checkpoint: str = PRETRAIN_CHECKPOINT
    tune_checkpoint: str = TUNE_CHECKPOINT


@dataclass
class MathLMConfig:
    base: Qwen3LMConfig = field(default_factory=Qwen3LMConfig)
    tune: MathTuneConfig = field(default_factory=MathTuneConfig)


def compose_math_lm_config(overrides: Sequence[str] | None = None) -> MathLMConfig:
    base = compose_qwen3_scratch_config(list(overrides or []))
    return MathLMConfig(base=base)


def math_dataset_root(cfg: MathLMConfig) -> pathlib.Path:
    return dataset_dir(cfg.base.data.dataset)


def load_corpus_train_text(cfg: MathLMConfig) -> str:
    root = math_dataset_root(cfg)
    train_path = root / "corpus_train.txt"
    if train_path.is_file():
        return train_path.read_text(encoding="utf-8")
    return (root / "corpus.txt").read_text(encoding="utf-8")


def equation_prompt_answer(row: dict) -> tuple[str, str]:
    lhs = Rational.from_pair(tuple(row["lhs"]))
    rhs = Rational.from_pair(tuple(row["rhs"]))
    result = Rational.from_pair(tuple(row["result"]))
    lhs_text = value_to_text(lhs, article=True)
    rhs_text = value_to_text(rhs, article=True)
    answer = value_to_text(result)
    phrase = OP_PHRASE[row["op"]]
    prompt = f"{lhs_text} {phrase} {rhs_text} equals"
    return prompt, answer


def build_qa_examples(rows: list[dict]) -> list[tuple[str, str]]:
    return [equation_prompt_answer(row) for row in rows]


class MathQADataset(Dataset):
    def __init__(self, examples: list[tuple[str, str]], tokenizer: Tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt, answer = self.examples[index]
        prompt_ids = self.tokenizer.encode_chars(prompt)
        answer_ids = self.tokenizer.encode_chars(answer)
        eos = torch.tensor([self.tokenizer.eos], dtype=torch.long)
        seq = torch.cat([prompt_ids, answer_ids, eos])
        input_ids = seq[:-1]
        labels = seq[1:].clone()
        prompt_len = len(prompt_ids)
        labels[: prompt_len - 1] = IGNORE_INDEX
        return input_ids, labels


def collate_math_qa(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item[0]) for item in batch)
    input_ids: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    attention: list[torch.Tensor] = []
    for inp, lab in batch:
        pad_len = max_len - len(inp)
        input_ids.append(F.pad(inp, (0, pad_len), value=pad_id))
        labels.append(F.pad(lab, (0, pad_len), value=IGNORE_INDEX))
        attention.append(
            torch.cat([torch.ones(len(inp), dtype=torch.float32), torch.zeros(pad_len)])
        )
    return (
        torch.stack(input_ids),
        torch.stack(labels),
        torch.stack(attention),
    )


class MathTrainCB(TrainCB):
    def predict(self, learn):
        batch = learn.batch
        if len(batch) >= 3:
            return learn.model(batch[0], attention_mask=batch[2])
        return learn.model(batch[0])


def create_math_tune_dls(
    cfg: MathLMConfig,
    *,
    tokenizer: Tokenizer,
    train_rows: list[dict],
    valid_rows: list[dict] | None = None,
) -> tuple[Learner, Tokenizer]:
    train_examples = build_qa_examples(train_rows)
    valid_examples = build_qa_examples(valid_rows or train_rows[: max(1, len(train_rows) // 20)])
    pad_id = tokenizer.pad
    collate = partial(collate_math_qa, pad_id=pad_id)
    loader_kw = {
        "batch_size": cfg.base.data.batch_size,
        "num_workers": cfg.base.data.num_workers,
        "pin_memory": cfg.base.data.pin_memory,
        "collate_fn": collate,
    }
    train_dl = DataLoader(MathQADataset(train_examples, tokenizer), shuffle=True, **loader_kw)
    valid_dl = DataLoader(MathQADataset(valid_examples, tokenizer), shuffle=False, **loader_kw)

    dls = Qwen3DataLoaders(train=train_dl, valid=valid_dl)
    model_cfg = replace(
        cfg.base.model,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max(cfg.base.model.max_position_embeddings, cfg.base.data.block_size),
    )
    model = Qwen3ForCausalLM(model_cfg)
    cbs = default_cbs(
        train=MathTrainCB(),
        compile=cfg.base.fit.compile,
        compile_mode=cfg.base.fit.compile_mode,
        channels_last=False,
        plot_progress=False,
        grad_accum=cfg.base.fit.grad_accum,
    )
    learn = Learner(
        model,
        dls,
        loss_func=partial(causal_lm_loss, ignore_index=IGNORE_INDEX),
        opt_func=partial(torch.optim.AdamW, weight_decay=cfg.base.fit.weight_decay),
        lr=cfg.base.fit.lr,
        cbs=cbs,
        pin_memory=cfg.base.data.pin_memory,
        project_name=cfg.base.fit.project_name or DEFAULT_PROJECT,
    )
    learn.tokenizer = tokenizer
    learn.cfg = cfg
    return learn, tokenizer


def default_pretrain_overrides() -> list[str]:
    return [
        "data.dataset=small",
        "data.text_file=corpus_train.txt",
        "data.block_size=64",
        "data.batch_size=64",
        "fit.epochs=15",
        "fit.lr=3e-4",
        f"fit.checkpoint={PRETRAIN_CHECKPOINT}",
    ]


def default_tune_overrides() -> list[str]:
    return [
        "data.dataset=small",
        "data.block_size=96",
        "data.batch_size=32",
        "fit.epochs=8",
        "fit.lr=1e-4",
        f"fit.checkpoint={TUNE_CHECKPOINT}",
    ]


def pretrain(cfg: MathLMConfig | None = None, *, plot_progress: bool = False) -> Learner:
    cfg = cfg or compose_math_lm_config(default_pretrain_overrides())
    text = load_corpus_train_text(cfg)
    dls, tokenizer = create_qwen3_dls(cfg.base, text=text)
    model_cfg = replace(
        cfg.base.model,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max(cfg.base.model.max_position_embeddings, cfg.base.data.block_size),
    )
    model = Qwen3ForCausalLM(model_cfg)
    cbs = default_cbs(
        train=TrainCB(),
        compile=cfg.base.fit.compile,
        compile_mode=cfg.base.fit.compile_mode,
        channels_last=False,
        plot_progress=plot_progress,
        grad_accum=cfg.base.fit.grad_accum,
    )
    learn = Learner(
        model,
        dls,
        loss_func=causal_lm_loss,
        opt_func=partial(torch.optim.AdamW, weight_decay=cfg.base.fit.weight_decay),
        lr=cfg.base.fit.lr,
        cbs=cbs,
        pin_memory=cfg.base.data.pin_memory,
        project_name=cfg.base.fit.project_name or DEFAULT_PROJECT,
    )
    learn.tokenizer = tokenizer
    learn.cfg = cfg
    learn.fit(cfg.base.fit.epochs)
    ckpt = save_qwen3_checkpoint(learn, cfg.base)
    learn.checkpoint_path = str(ckpt)
    print(f"pretrain checkpoint: {ckpt}")
    return learn


def tune(
    cfg: MathLMConfig | None = None,
    *,
    checkpoint: str | pathlib.Path | None = None,
    plot_progress: bool = False,
) -> Learner:
    cfg = cfg or compose_math_lm_config(default_tune_overrides())
    ckpt_path = pathlib.Path(checkpoint) if checkpoint is not None else pathlib.Path(
        checkpoint_path(cfg.base.fit.project_name or DEFAULT_PROJECT, cfg.tune.pretrain_checkpoint)
    )
    model, tokenizer, loaded_cfg, _ = load_qwen3_checkpoint(ckpt_path)
    tune_ckpt = cfg.base.fit.checkpoint
    tune_epochs = cfg.base.fit.epochs
    tune_lr = cfg.base.fit.lr
    cfg.base = loaded_cfg
    cfg.base.fit.checkpoint = tune_ckpt
    cfg.base.fit.epochs = tune_epochs
    cfg.base.fit.lr = tune_lr

    train_rows, holdout_rows = load_equations_split(math_dataset_root(cfg))
    learn, _ = create_math_tune_dls(cfg, tokenizer=tokenizer, train_rows=train_rows, valid_rows=holdout_rows)
    learn.model.load_state_dict(model.state_dict())
    learn.fit(cfg.base.fit.epochs)
    ckpt = save_qwen3_checkpoint(learn, cfg.base)
    learn.checkpoint_path = str(ckpt)
    print(f"tune checkpoint: {ckpt}")
    return learn


@torch.no_grad()
def generate_answer(
    model: Qwen3ForCausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
) -> str:
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode_chars(prompt).unsqueeze(0).to(device)
    out = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        greedy=True,
        eos_token_id=tokenizer.eos,
        top_k=None,
    )
    generated = out[0, prompt_ids.shape[1] :]
    return tokenizer.decode(generated).strip()


def evaluate_math_lm(
    model: Qwen3ForCausalLM,
    tokenizer: Tokenizer,
    rows: list[dict],
    *,
    max_new_tokens: int = 64,
) -> dict:
    model.eval()
    total = 0
    correct = 0
    by_op: dict[str, list[bool]] = defaultdict(list)
    by_domain: dict[str, list[bool]] = defaultdict(list)
    samples: list[dict] = []

    for row in rows:
        prompt, expected = equation_prompt_answer(row)
        predicted = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        ok = predicted == expected
        total += 1
        correct += int(ok)
        by_op[row["op"]].append(ok)
        by_domain[row["domain"]].append(ok)
        if len(samples) < 5:
            samples.append({"prompt": prompt, "expected": expected, "predicted": predicted, "ok": ok})

    def _acc(items: list[bool]) -> float:
        return sum(items) / len(items) if items else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_op": {op: _acc(vals) for op, vals in by_op.items()},
        "by_domain": {dom: _acc(vals) for dom, vals in by_domain.items()},
        "samples": samples,
    }


def eval_checkpoint(
    checkpoint: str | pathlib.Path,
    *,
    split: str = "holdout",
    max_new_tokens: int = 64,
) -> dict:
    model, tokenizer, _, _ = load_qwen3_checkpoint(checkpoint)
    train_rows, holdout_rows = load_equations_split(default_dataset_dir())
    rows = holdout_rows if split == "holdout" else train_rows
    metrics = evaluate_math_lm(model, tokenizer, rows, max_new_tokens=max_new_tokens)
    print(json.dumps({k: v for k, v in metrics.items() if k != "samples"}, indent=2))
    for sample in metrics["samples"]:
        mark = "ok" if sample["ok"] else "miss"
        print(f"[{mark}] {sample['prompt']} -> expected={sample['expected']!r} got={sample['predicted']!r}")
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Two-step math char LM training")
    parser.add_argument("--stage", choices=("pretrain", "tune", "eval"), required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--split", choices=("holdout", "train"), default="holdout")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides: list[str] = []
    if args.stage == "pretrain":
        overrides = default_pretrain_overrides()
    elif args.stage == "tune":
        overrides = default_tune_overrides()
    if args.epochs is not None:
        overrides.append(f"fit.epochs={args.epochs}")
    if args.lr is not None:
        overrides.append(f"fit.lr={args.lr}")

    if args.stage == "pretrain":
        pretrain(compose_math_lm_config(overrides), plot_progress=args.plot)
        return

    if args.stage == "tune":
        tune(compose_math_lm_config(overrides), checkpoint=args.checkpoint, plot_progress=args.plot)
        return

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = pathlib.Path(
            checkpoint_path(DEFAULT_PROJECT, TUNE_CHECKPOINT)
        )
        if not ckpt.is_file():
            ckpt = pathlib.Path(checkpoint_path(DEFAULT_PROJECT, PRETRAIN_CHECKPOINT))
    eval_checkpoint(ckpt, split=args.split)


if __name__ == "__main__":
    main()
