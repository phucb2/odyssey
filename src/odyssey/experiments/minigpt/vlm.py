"""MiniGPT-style VLM: HF CLIP vision + scratch Q-Former + Qwen3 LM.

Frozen CLIP and Qwen3; trainable Q-Former and linear projector map image
features into soft prompts prepended to text token embeddings.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModel, GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from odyssey.experiments.minigpt.qformer import QFormer, QFormerConfig
from odyssey.experiments.minigpt.qwen3 import Qwen3LM, model_name as default_llm_name

DEFAULT_CLIP_NAME = "openai/clip-vit-base-patch32"


def _hf_logits_processors(generation_config: GenerationConfig) -> LogitsProcessorList:
    processors = LogitsProcessorList()
    if not generation_config.do_sample:
        return processors

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


@dataclass
class MiniGPTConfig:
    """Top-level config for the VLM stack."""

    clip_model_name: str = DEFAULT_CLIP_NAME
    llm_model_name: str = default_llm_name
    num_queries: int = 32
    qformer_layers: int = 6
    qformer_heads: int = 12
    qformer_ffn_dim: int = 3072
    cross_attention_freq: int = 1


class HFCLIPVision(nn.Module):
    """Thin wrapper around a frozen HuggingFace CLIP vision tower."""

    def __init__(self, model_name: str = DEFAULT_CLIP_NAME):
        super().__init__()
        self.model_name = model_name
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self._image_processor = CLIPImageProcessor.from_pretrained(model_name)

    @property
    def image_processor(self) -> CLIPImageProcessor:
        return self._image_processor

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return patch token features: (batch, num_patches, hidden_size)."""
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return outputs.last_hidden_state

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False


class VisionLanguageProjector(nn.Module):
    """Linear map from Q-Former dim to LLM hidden size."""

    def __init__(self, qformer_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Linear(qformer_dim, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MiniGPT(nn.Module):
    """CLIP + Q-Former + projector + Qwen3 causal LM."""

    def __init__(
        self,
        cfg: MiniGPTConfig,
        llm: Qwen3LM,
        *,
        generation_config: GenerationConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.llm = llm
        self.generation_config = generation_config or copy.deepcopy(llm.generation_config)

        self.vision = HFCLIPVision(cfg.clip_model_name)
        qformer_cfg = QFormerConfig(
            num_queries=cfg.num_queries,
            hidden_size=self.vision.hidden_size,
            num_heads=cfg.qformer_heads,
            num_layers=cfg.qformer_layers,
            encoder_hidden_size=self.vision.hidden_size,
            ffn_dim=cfg.qformer_ffn_dim,
            cross_attention_freq=cfg.cross_attention_freq,
        )
        self.qformer = QFormer(qformer_cfg)
        self.projector = VisionLanguageProjector(self.vision.hidden_size, llm.cfg.hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        cfg: MiniGPTConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = torch.float32,
    ) -> MiniGPT:
        cfg = cfg or MiniGPTConfig()
        llm = Qwen3LM.from_pretrained(cfg.llm_model_name, device=device, dtype=dtype)
        gen_cfg = GenerationConfig.from_pretrained(cfg.llm_model_name)
        model = cls(cfg, llm, generation_config=gen_cfg)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model

    @property
    def num_visual_tokens(self) -> int:
        return self.qformer.num_queries

    def freeze_backbone(self) -> None:
        """Freeze CLIP and LLM; leave Q-Former + projector trainable."""
        self.vision.freeze()
        for param in self.llm.parameters():
            param.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Map images to LLM soft prompts: (batch, num_queries, llm_hidden)."""
        image_feats = self.vision.encode(pixel_values)
        query_feats = self.qformer(image_feats)
        return self.projector(query_feats)

    def _build_inputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_embeds = self.encode_image(pixel_values)
        text_embeds = self.llm.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        batch, num_visual = visual_embeds.shape[:2]
        if attention_mask is None:
            text_mask = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.long)
        else:
            text_mask = attention_mask
        visual_mask = torch.ones(batch, num_visual, device=input_ids.device, dtype=text_mask.dtype)
        full_mask = torch.cat([visual_mask, text_mask], dim=1)
        return inputs_embeds, full_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        inputs_embeds, full_mask = self._build_inputs(pixel_values, input_ids, attention_mask)
        logits = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask)

        loss = None
        if labels is not None:
            num_visual = self.num_visual_tokens
            ignore = torch.full(
                (labels.shape[0], num_visual),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            full_labels = torch.cat([ignore, labels], dim=1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def _resolve_generation_config(self, **kwargs) -> GenerationConfig:
        gen_cfg = copy.deepcopy(self.generation_config)
        if kwargs.get("max_new_tokens") is not None:
            gen_cfg.max_length = None
        gen_cfg.update(**{k: v for k, v in kwargs.items() if v is not None})
        return gen_cfg

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive decode with a fixed visual prefix."""
        gen_cfg = self._resolve_generation_config(**kwargs)
        processors = _hf_logits_processors(gen_cfg)
        eos_ids = set(_eos_ids(gen_cfg))

        max_new_tokens = gen_cfg.max_new_tokens
        if max_new_tokens is None:
            max_length = gen_cfg.max_length or (input_ids.shape[1] + 20)
            max_new_tokens = max(0, max_length - input_ids.shape[1])

        self.eval()
        inputs_embeds, full_mask = self._build_inputs(pixel_values, input_ids, attention_mask)
        generated_ids: list[torch.Tensor] = []
        token_ids = input_ids
        unfinished = torch.ones(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)
        block = self.llm.cfg.max_position_embeddings

        for _ in range(int(max_new_tokens)):
            ctx_embeds = inputs_embeds[:, -block:]
            ctx_mask = full_mask[:, -block:]
            logits = self.llm(inputs_embeds=ctx_embeds, attention_mask=ctx_mask)[:, -1, :]
            logits = processors(token_ids, logits)

            if gen_cfg.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)

            generated_ids.append(next_id)
            token_ids = torch.cat([token_ids, next_id], dim=1)

            if eos_ids and gen_cfg.pad_token_id is not None:
                next_id = torch.where(
                    unfinished[:, None],
                    next_id,
                    torch.full_like(next_id, int(gen_cfg.pad_token_id)),
                )

            next_embed = self.llm.model.embed_tokens(next_id)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            full_mask = torch.cat(
                [full_mask, unfinished[:, None].to(full_mask.dtype)],
                dim=1,
            )

            if eos_ids:
                unfinished = unfinished & ~torch.isin(
                    next_id.squeeze(-1),
                    torch.tensor(list(eos_ids), device=inputs_embeds.device),
                )
                if not unfinished.any():
                    break

        if not generated_ids:
            return input_ids
        new_ids = torch.cat(generated_ids, dim=1)
        return torch.cat([input_ids, new_ids], dim=1)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict for Q-Former + projector only."""
        state = {}
        for key, value in self.qformer.state_dict().items():
            state[f"qformer.{key}"] = value
        for key, value in self.projector.state_dict().items():
            state[f"projector.{key}"] = value
        return state

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        qformer_state = {k.removeprefix("qformer."): v for k, v in state.items() if k.startswith("qformer.")}
        proj_state = {k.removeprefix("projector."): v for k, v in state.items() if k.startswith("projector.")}
        if qformer_state:
            self.qformer.load_state_dict(qformer_state)
        if proj_state:
            self.projector.load_state_dict(proj_state)
