# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Non-intrusive Multi-Token Prediction (MTP) helpers for SFT.

These utilities allow training MTP layers without modifying transformers model
files. They are used by `CustomSeq2SeqTrainer` when `num_mtp_layers > 0`.
"""

from typing import Any

import torch
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import MtpModel


def extend_layer_types(config: Any, num_mtp_layers: int) -> Any:
    r"""Extend `layer_types` so that MTP layers can be instantiated.

    Models such as Qwen3.5 read `config.layer_types[layer_idx]` in their decoder
    layer. MTP layers have `layer_idx = num_hidden_layers + k`, so we append
    extra `"full_attention"` entries at runtime without touching any file.
    """
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    current = list(getattr(text_config, "layer_types", []))
    needed = text_config.num_hidden_layers + num_mtp_layers
    if len(current) < needed:
        current.extend(["full_attention"] * (needed - len(current)))
        text_config.layer_types = current
    return text_config


def get_base_lm_model(model: torch.nn.Module) -> torch.nn.Module:
    r"""Return the underlying language model (text tower).

    This handles PeftModel wrapping and multimodal models:
      - PeftModel -> get_base_model()
      - Conditional generation -> base_model.language_model
      - Causal LM -> base_model
    """
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()

    base = getattr(model, "base_model", model)
    return getattr(base, "language_model", base)


def compute_mtp_loss_tf(
    mtp_model: MtpModel,
    input_ids: torch.Tensor,
    main_hidden_states: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Teacher-forcing MTP loss computed entirely outside the base model.

    Layer i consumes the embedding of token ``t+i+1`` and the previous hidden
    state at position ``t`` to predict token ``t+i+2``.
    """
    batch_size, seq_len = main_hidden_states.shape[:2]
    device = main_hidden_states.device

    base_embeds = mtp_model.embed_tokens(input_ids)
    pos = torch.arange(seq_len, device=device).view(1, 1, -1)
    base_position_ids = pos.expand(4, batch_size, -1).clone()

    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    current_hidden = main_hidden_states

    for i in range(len(mtp_model.layers)):
        shift = i + 1
        shifted_embeds = base_embeds[:, shift:, :]
        shifted_embeds = F.pad(shifted_embeds, (0, 0, 0, shift), value=0.0)

        shifted_position_ids = torch.roll(base_position_ids, -shift, dims=-1).clone()
        shifted_position_ids[..., -shift:] = base_position_ids[..., -shift:]

        # Qwen3.5 rotary_emb expects mrope (3D) for temporal/height/width.
        position_embeddings = mtp_model.rotary_emb(shifted_embeds, position_ids=shifted_position_ids[1:])

        # Only shift 2D attention masks; for 4D or None fall back to a pure causal mask.
        # Labels are -100 for padded positions so the loss ignores them.
        if attention_mask is not None and attention_mask.dim() == 2:
            shifted_attention_mask = attention_mask[:, shift:]
            shifted_attention_mask = F.pad(shifted_attention_mask, (0, shift), value=0)
        else:
            shifted_attention_mask = None

        causal_mask = create_causal_mask(
            config=mtp_model.config,
            inputs_embeds=shifted_embeds,
            attention_mask=shifted_attention_mask,
            past_key_values=None,
            position_ids=shifted_position_ids[0],
        )

        current_hidden = mtp_model.layers[i](
            inputs_embeds=shifted_embeds,
            previous_hidden_state=current_hidden,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=shifted_position_ids[0],
            past_key_values=None,
        )

        mtp_logits = mtp_model.shared_head(current_hidden)

        target_shift = i + 2
        shifted_labels = torch.roll(labels, -target_shift, dims=1).clone()
        shifted_labels[:, -target_shift:] = -100

        layer_loss = mtp_model.loss_function(
            mtp_logits,
            labels,
            vocab_size=mtp_model.config.vocab_size,
            shift_labels=shifted_labels,
        )
        total_loss = total_loss + layer_loss

    return total_loss / len(mtp_model.layers)


class MtpHiddenStateHook:
    r"""Forward hook that captures the output of the base model's final norm."""

    def __init__(self, module: torch.nn.Module):
        self.hidden: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.hidden = output if isinstance(output, torch.Tensor) else output[0]

    def remove(self):
        self.handle.remove()
