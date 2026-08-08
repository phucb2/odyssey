"""Load pretrained Qwen3 weights into the scratch decoder and run generate.

Educational script: maps HF ``Qwen/Qwen3-*`` config + state_dict onto
``Qwen3ForCausalLM`` from ``qwen3_scatch``, then samples like transformers.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from odyssey.experiments.minigpt.qwen3_scatch import Qwen3ForCausalLM, Qwen3ScratchConfig

model_name = "Qwen/Qwen3-0.6B"


def _rope_theta(cfg) -> float:
    """HF moved ``rope_theta`` into ``rope_parameters`` on recent Qwen3 configs."""
    if hasattr(cfg, "rope_theta"):
        return float(cfg.rope_theta)
    params = getattr(cfg, "rope_parameters", None) or {}
    return float(params.get("rope_theta", 10000.0))


def from_qwen_cfg(cfg) -> Qwen3ScratchConfig:
    """Convert a transformers Qwen3 config to ``Qwen3ScratchConfig``."""
    return Qwen3ScratchConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=_rope_theta(cfg),
        attention_bias=cfg.attention_bias,
        tie_word_embeddings=cfg.tie_word_embeddings,
    )


def _hf_logits_processors(generation_config: GenerationConfig) -> LogitsProcessorList:
    """Build the same sampling warpers HF uses in ``_get_logits_processor`` (sample path)."""
    processors = LogitsProcessorList()
    if not generation_config.do_sample:
        return processors

    # HF keeps min_tokens_to_keep=1 for non-beam sampling.
    min_tokens_to_keep = 1
    if generation_config.temperature is not None and generation_config.temperature != 1.0:
        processors.append(TemperatureLogitsWarper(generation_config.temperature))
    if generation_config.top_k is not None and generation_config.top_k != 0:
        processors.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
    if generation_config.top_p is not None and generation_config.top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
    return processors


def _eos_ids(generation_config: GenerationConfig) -> list[int]:
    eos = generation_config.eos_token_id
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return list(eos)


class Qwen3LM(Qwen3ForCausalLM):
    """Scratch causal LM that loads HuggingFace Qwen3 pretrained weights."""

    def __init__(
        self,
        cfg: Qwen3ScratchConfig,
        generation_config: GenerationConfig | None = None,
    ):
        super().__init__(cfg)
        self.generation_config = generation_config or GenerationConfig()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = model_name,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = torch.float32,
    ) -> Qwen3LM:
        hf_cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        scratch_cfg = from_qwen_cfg(hf_cfg)
        gen_cfg = GenerationConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(scratch_cfg, generation_config=gen_cfg)

        hf_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        missing, unexpected = model.load_state_dict(hf_model.state_dict(), strict=True)
        del hf_model
        if missing or unexpected:
            raise RuntimeError(f"weight load mismatch: missing={missing} unexpected={unexpected}")

        if device is not None:
            model = model.to(device)
        model.eval()
        return model

    def _resolve_generation_config(self, **kwargs) -> GenerationConfig:
        """Merge call-site kwargs into a copy of the model GenerationConfig (HF-style)."""
        gen_cfg = copy.deepcopy(self.generation_config)
        # Prefer max_new_tokens over stale max_length from the saved config.
        if kwargs.get("max_new_tokens") is not None:
            gen_cfg.max_length = None
        gen_cfg.update(**{k: v for k, v in kwargs.items() if v is not None})
        return gen_cfg

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """HF-compatible generate using the same GenerationConfig + warper strategy.

        Qwen3 defaults: ``do_sample=True``, ``temperature=0.6``, ``top_k=20``, ``top_p=0.95``.
        Call-site kwargs override those defaults, same as ``transformers``.
        """
        if input_ids is None:
            raise ValueError("input_ids is required")

        gen_cfg = self._resolve_generation_config(**kwargs)
        processors = _hf_logits_processors(gen_cfg)
        eos_ids = set(_eos_ids(gen_cfg))

        max_new_tokens = gen_cfg.max_new_tokens
        if max_new_tokens is None:
            # HF falls back to max_length - prompt_len when max_new_tokens is unset.
            max_length = gen_cfg.max_length or (input_ids.shape[1] + 20)
            max_new_tokens = max(0, max_length - input_ids.shape[1])

        self.eval()
        ids = input_ids
        block = self.cfg.max_position_embeddings
        unfinished = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)

        for _ in range(int(max_new_tokens)):
            ctx = ids[:, -block:]
            mask = None
            if attention_mask is not None:
                mask = attention_mask[:, -block:]
            logits = self(ctx, attention_mask=mask)[:, -1, :]
            logits = processors(ids, logits)

            if gen_cfg.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)

            # Finished sequences keep emitting pad (HF behaviour with eos stopping).
            if eos_ids and gen_cfg.pad_token_id is not None:
                next_id = torch.where(
                    unfinished[:, None],
                    next_id,
                    torch.full_like(next_id, int(gen_cfg.pad_token_id)),
                )

            ids = torch.cat([ids, next_id], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, unfinished[:, None].to(attention_mask.dtype)],
                    dim=1,
                )

            if eos_ids:
                unfinished = unfinished & ~torch.isin(next_id.squeeze(-1), torch.tensor(list(eos_ids), device=ids.device))
                if not unfinished.any():
                    break

        return ids


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3LM.from_pretrained(model_name, device=device)

    # prepare the model input
    prompt = "What is ballon?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    # conduct text completion
    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)
