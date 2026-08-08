"""Scratch Qwen3-style decoder + char LM training on Truyện Kiều.

Educational reimplementation for experiments — not a full HF port.
Train via ``python -m odyssey.experiments.minigpt.qwen3_scatch`` (clip-style layout).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from odyssey.paths import DEFAULT_PROJECT, checkpoint_path, dataset_dir
from odyssey.training.callbacks import TrainCB, default_cbs, make_lr_find
from odyssey.training.learner import Learner

_QWEN3_SCRATCH_CONFIG_REGISTERED = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TokenizerConfig:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2


@dataclass
class DataConfig:
    dataset: str = "trieukieu"
    text_file: str = "truyenkieu.txt"
    batch_size: int = 64
    block_size: int = 32
    valid_frac: float = 0.1
    seed: int = 1337
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class Qwen3ScratchConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 16
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True


@dataclass
class FitConfig:
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_accum: int = 1
    compile: bool = False
    compile_mode: str = "default"
    project_name: str = "odyssey"
    checkpoint: str = "qwen3_scratch_truyenkieu.pt"


@dataclass
class Qwen3LMConfig:
    """Nested experiment config (Hydra / OmegaConf structured)."""

    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: Qwen3ScratchConfig = field(default_factory=Qwen3ScratchConfig)
    fit: FitConfig = field(default_factory=FitConfig)


def register_qwen3_scratch_config() -> None:
    global _QWEN3_SCRATCH_CONFIG_REGISTERED
    if _QWEN3_SCRATCH_CONFIG_REGISTERED:
        return
    ConfigStore.instance().store(name="qwen3_scratch_config", node=Qwen3LMConfig)
    _QWEN3_SCRATCH_CONFIG_REGISTERED = True


register_qwen3_scratch_config()


def compose_qwen3_scratch_config(overrides: Sequence[str] | None = None) -> Qwen3LMConfig:
    """Compose Qwen3LMConfig from structured defaults plus Hydra-style overrides."""
    register_qwen3_scratch_config()
    with initialize_config_module(config_module="odyssey.conf", version_base="1.3"):
        cfg = compose(config_name="qwen3_scratch", overrides=list(overrides or []))
    return OmegaConf.to_object(cfg)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class Tokenizer:
    """Char-level tokenizer: pad=_=0, bos=^=1, eos=$=2."""

    PAD_CHAR, BOS_CHAR, EOS_CHAR = "_", "^", "$"

    def __init__(self, cfg: TokenizerConfig | None = None):
        cfg = cfg or TokenizerConfig()
        self.pad, self.bos, self.eos = cfg.pad_id, cfg.bos_id, cfg.eos_id
        self.stoi: dict[str, int] = {
            self.PAD_CHAR: self.pad,
            self.BOS_CHAR: self.bos,
            self.EOS_CHAR: self.eos,
        }
        self.itos: dict[int, str] = {v: k for k, v in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def fit(self, texts: list[str] | str) -> None:
        if isinstance(texts, str):
            texts = [texts]
        chars = sorted({c for text in texts for c in text if c not in self.stoi})
        next_id = max(self.stoi.values()) + 1
        for ch in chars:
            self.stoi[ch] = next_id
            self.itos[next_id] = ch
            next_id += 1

    def encode(self, text: str) -> torch.Tensor:
        """Encode with BOS/EOS wrappers (CLIP-style sequences)."""
        try:
            ids = [self.bos] + [self.stoi[c] for c in text] + [self.eos]
        except KeyError as e:
            raise ValueError(f"unknown character {e.args[0]!r}; call fit() first") from e
        return torch.tensor(ids, dtype=torch.long)

    def encode_chars(self, text: str) -> torch.Tensor:
        """Encode raw characters with no BOS/EOS (continuous LM corpus)."""
        try:
            ids = [self.stoi[c] for c in text]
        except KeyError as e:
            raise ValueError(f"unknown character {e.args[0]!r}; call fit() first") from e
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        skip = {self.pad, self.bos, self.eos}
        return "".join(self.itos[i] for i in ids.tolist() if i not in skip)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def resolve_text_path(cfg: DataConfig) -> pathlib.Path:
    """Resolve ``datasets/<dataset>/<text_file>`` (defaults to Truyện Kiều)."""
    path = dataset_dir(cfg.dataset) / cfg.text_file
    if not path.is_file():
        # Tolerate common typo from older notes.
        alt = dataset_dir(cfg.dataset) / "trienkieu.txt"
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"corpus not found: {path}")
    return path


def load_corpus(cfg: DataConfig | None = None) -> str:
    cfg = cfg or DataConfig()
    return resolve_text_path(cfg).read_text(encoding="utf-8")


class CharLMDataset(Dataset):
    """Sliding windows over a 1D token tensor: x[i:i+T], y[i+1:i+T+1]."""

    def __init__(self, data: torch.Tensor, block_size: int):
        if data.ndim != 1:
            raise ValueError(f"expected 1D token tensor, got shape {tuple(data.shape)}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if len(data) <= block_size:
            raise ValueError(f"data length {len(data)} must exceed block_size {block_size}")
        self.data = data.long()
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(index)
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + self.block_size + 1]
        return x, y


@dataclass
class Qwen3DataLoaders:
    train: DataLoader
    valid: DataLoader


def build_tokenizer_and_data(
    cfg: Qwen3LMConfig,
    *,
    text: str | None = None,
) -> tuple[Tokenizer, torch.Tensor, torch.Tensor]:
    """Fit tokenizer on the corpus and split into contiguous train/valid id tensors."""
    text = text if text is not None else load_corpus(cfg.data)
    tokenizer = Tokenizer(cfg.tokenizer)
    tokenizer.fit(text)
    data = tokenizer.encode_chars(text)

    torch.manual_seed(cfg.data.seed)
    n = len(data)
    n_valid = max(1, int(round(n * cfg.data.valid_frac)))
    n_train = n - n_valid
    if n_train <= cfg.data.block_size:
        raise ValueError(
            f"train tokens ({n_train}) must exceed block_size ({cfg.data.block_size})"
        )
    train_data = data[:n_train]
    valid_data = data[n_train:]
    if len(valid_data) <= cfg.data.block_size:
        # Fall back: keep a tiny valid split by borrowing from train tail.
        valid_data = data[-(cfg.data.block_size + 1) :]
    return tokenizer, train_data, valid_data


def get_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Karpathy-style random batch of (x, y) windows from a 1D token tensor."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def create_qwen3_dls(
    cfg: Qwen3LMConfig,
    *,
    text: str | None = None,
) -> tuple[Qwen3DataLoaders, Tokenizer]:
    """Build train/valid loaders over char-LM windows."""
    tokenizer, train_data, valid_data = build_tokenizer_and_data(cfg, text=text)
    train_ds = CharLMDataset(train_data, cfg.data.block_size)
    valid_ds = CharLMDataset(valid_data, cfg.data.block_size)
    loader_kw = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory,
    }
    dls = Qwen3DataLoaders(
        train=DataLoader(train_ds, shuffle=True, **loader_kw),
        valid=DataLoader(valid_ds, shuffle=False, **loader_kw),
    )
    return dls, tokenizer


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    def __init__(self, base: float, dim: int):
        super().__init__()
        self.base, self.dim = base, dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: (..., seq_len)
        returns: cos, sin each (..., seq_len, dim)
        """
        inv_freq = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        freqs = torch.einsum("...i,j->...ij", position_ids.to(torch.float32), inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: (batch, n_heads, seq_len, head_dim)
    cos, sin: (batch, seq_len, head_dim) — unsqueezed along `unsqueeze_dim` for broadcast.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    x: (batch, n_kv_heads, seq_len, head_dim)
    returns: (batch, n_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


class QwenAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
        causal: bool = True,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = self.head_dim**-0.5
        self.num_kv_groups = num_heads // num_kv_heads
        self.causal = causal

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        q = self.q_proj(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_weights = q @ k.transpose(-2, -1) * self.scaling

        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
        if attention_mask is not None:
            padding_mask = (1.0 - attention_mask[:, None, None, :].to(attn_weights.dtype)) * torch.finfo(
                attn_weights.dtype
            ).min
            attn_weights = attn_weights + padding_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3ScratchConfig):
        super().__init__()
        self.self_attn = QwenAttention(
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            bias=cfg.attention_bias,
            causal=True,
            rms_norm_eps=cfg.rms_norm_eps,
        )
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, attention_mask)
        x = x + residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3ScratchConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = RotaryEmbedding(cfg.rope_theta, cfg.head_dim)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device)
            position_ids = position_ids.expand(x.shape[0], -1)
        cos, sin = self.rope(position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    """Decoder + tied/untied LM head; forward returns next-token logits."""

    def __init__(self, cfg: Qwen3ScratchConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.model(input_ids, position_ids=position_ids, attention_mask=attention_mask)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        """Greedy / multinomial generate from a prompt batch of shape (B, T)."""
        self.eval()
        ids = input_ids
        block = self.cfg.max_position_embeddings
        for _ in range(max_new_tokens):
            ctx = ids[:, -block:]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Token-level cross-entropy over a (B, T, V) logit tensor."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


# ---------------------------------------------------------------------------
# Training (clip.py layout)
# ---------------------------------------------------------------------------


class Qwen3TrainCB(TrainCB):
    """Standard (x, y) LM batch; TrainCB already matches this shape."""


def build_qwen3_learner(cfg: Qwen3LMConfig, *, plot_progress: bool = True) -> Learner:
    """Construct a Learner for scratch Qwen3 char LM."""
    dls, tokenizer = create_qwen3_dls(cfg)
    model_cfg = replace(
        cfg.model,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max(cfg.model.max_position_embeddings, cfg.data.block_size),
    )
    model = Qwen3ForCausalLM(model_cfg)

    cbs = default_cbs(
        train=Qwen3TrainCB(),
        compile=cfg.fit.compile,
        compile_mode=cfg.fit.compile_mode,
        channels_last=False,
        plot_progress=plot_progress,
        grad_accum=cfg.fit.grad_accum,
    )
    opt_func = partial(torch.optim.AdamW, weight_decay=cfg.fit.weight_decay)
    learn = Learner(
        model,
        dls,
        loss_func=causal_lm_loss,
        opt_func=opt_func,
        lr=cfg.fit.lr,
        cbs=cbs,
        lr_find=make_lr_find(show_plot=plot_progress),
        pin_memory=cfg.data.pin_memory,
        project_name=cfg.fit.project_name or DEFAULT_PROJECT,
    )
    learn.tokenizer = tokenizer
    learn.cfg = cfg
    return learn


def save_qwen3_checkpoint(
    learn: Learner,
    cfg: Qwen3LMConfig,
    *,
    path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Save model, tokenizer vocab, and config after training."""
    project = cfg.fit.project_name or DEFAULT_PROJECT
    out = pathlib.Path(path) if path is not None else pathlib.Path(
        checkpoint_path(project, cfg.fit.checkpoint)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer: Tokenizer = learn.tokenizer
    payload = {
        "model": learn.model.state_dict(),
        "tokenizer": {
            "stoi": tokenizer.stoi,
            "itos": {int(k): v for k, v in tokenizer.itos.items()},
            "pad": tokenizer.pad,
            "bos": tokenizer.bos,
            "eos": tokenizer.eos,
        },
        "cfg": OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
        "model_cfg": OmegaConf.to_container(
            OmegaConf.structured(learn.model.cfg), resolve=True
        ),
    }
    torch.save(payload, out)
    return out


@torch.no_grad()
def sample_text(
    learn: Learner,
    prompt: str = "Trăm năm",
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Generate a continuation from ``prompt`` using the fitted tokenizer."""
    model: Qwen3ForCausalLM = learn.model
    tokenizer: Tokenizer = learn.tokenizer
    device = next(model.parameters()).device
    ids = tokenizer.encode_chars(prompt).unsqueeze(0).to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].cpu())


def train(cfg: Qwen3LMConfig | None = None, *, plot_progress: bool = True) -> Learner:
    """Build the learner, fit, sample a short continuation, then checkpoint."""
    cfg = cfg or compose_qwen3_scratch_config()
    learn = build_qwen3_learner(cfg, plot_progress=plot_progress)
    learn.fit(cfg.fit.epochs)
    try:
        sample = sample_text(learn)
        print(f"sample:\n{sample}")
    except Exception as e:  # noqa: BLE001 — sampling is best-effort after fit
        print(f"sample failed: {e}")
    ckpt = save_qwen3_checkpoint(learn, cfg)
    learn.checkpoint_path = str(ckpt)
    print(f"checkpoint: {ckpt}")
    return learn


# Back-compat alias used by earlier notebook / demo snippets.
cfg = Qwen3ScratchConfig()


if __name__ == "__main__":
    import sys

    exp_cfg = compose_qwen3_scratch_config(sys.argv[1:])
    print(OmegaConf.to_yaml(OmegaConf.structured(exp_cfg)))
    text = load_corpus(exp_cfg.data)
    tok, train_data, valid_data = build_tokenizer_and_data(exp_cfg, text=text)
    print(
        f"corpus={resolve_text_path(exp_cfg.data)} chars={len(text)} "
        f"vocab={tok.vocab_size} train_tokens={len(train_data)} valid_tokens={len(valid_data)} "
        f"block_size={exp_cfg.data.block_size} batch_size={exp_cfg.data.batch_size}"
    )
    xb, yb = get_batch(
        train_data,
        batch_size=min(4, exp_cfg.data.batch_size),
        block_size=min(8, exp_cfg.data.block_size),
    )
    print(f"get_batch demo: xb={tuple(xb.shape)} yb={tuple(yb.shape)}")
    print("decoded text", tok.decode(xb[0]))
    print("decoded text", tok.decode(yb[0]))
    learn = train(exp_cfg, plot_progress=False)
    print(f"done: epochs={exp_cfg.fit.epochs} lr={exp_cfg.fit.lr}")
