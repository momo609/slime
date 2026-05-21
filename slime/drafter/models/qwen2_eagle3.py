"""
Eagle3 Model for Qwen2

This module implements the Eagle3 drafter model for Qwen2 architecture.
The Eagle3 model is a single-layer transformer that takes base model
hidden states as additional input to predict next tokens efficiently.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen2EagleDecoderLayer(nn.Module):
    """Eagle3 decoder layer for Qwen2.
    
    This layer takes both input embeddings and base model hidden states,
    concatenating them for attention computation. This allows the drafter
    to leverage the base model's representations for better prediction.
    """
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        
        # Attention layer with concatenated input
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        
        # Override QKV projections for Eagle3: input is [input_embeds, base_hidden_states]
        # So input dimension is hidden_size * 2
        self.self_attn.q_proj = nn.Linear(
            config.hidden_size * 2,
            config.num_attention_heads * self.head_dim,
            bias=True
        )
        self.self_attn.k_proj = nn.Linear(
            config.hidden_size * 2,
            config.num_key_value_heads * self.head_dim,
            bias=True
        )
        self.self_attn.v_proj = nn.Linear(
            config.hidden_size * 2,
            config.num_key_value_heads * self.head_dim,
            bias=True
        )
        
        # MLP layer
        self.mlp = Qwen2MLP(config)
        
        # Layer normalization
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Additional normalization for base hidden states
        self.hidden_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Forward pass of Eagle3 decoder layer.
        
        Args:
            input_embeds: Input embeddings from token ids [batch, seq_len, hidden_size]
            hidden_states: Base model hidden states [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Past key value cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to use cache
            cache_position: Cache position
            position_embeddings: Position embeddings
            
        Returns:
            Tuple of (hidden_states, present_key_value)
        """
        residual = hidden_states
        # print("############Qwen2EagleDecoderLayer hidden_states",hidden_states.shape)
        mid = hidden_states.shape[2] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden
        # print("############Qwen2EagleDecoderLayer embeds",embeds.shape)
        # Normalize inputs
        input_embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden)
        # print("############Qwen2EagleDecoderLayer input_embeds",input_embeds.shape)
        # print("############Qwen2EagleDecoderLayer hidden_states after",hidden_states.shape)
        
        # Concatenate input_embeds and hidden_states (key Eagle3 innovation)
        # This allows the drafter to use both token embeddings and base model representations
        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
        # print("############Qwen2EagleDecoderLayer hidden_states after2",hidden_states.shape)
        
        # Self-attention with concatenated input
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs


class Qwen2ForCausalLMEagle3(nn.Module):
    """Eagle3 drafter model for Qwen2.
    
    This is a lightweight single-layer transformer that serves as a drafter
    for speculative decoding. It takes base model hidden states as additional
    input to make fast token predictions.
    """
    
    def __init__(self, config: Qwen2Config):
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Embedding layer (shared with base model in practice)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Rotary embeddings
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        
        # Single decoder layer
        self.layers = nn.ModuleList([Qwen2EagleDecoderLayer(config, layer_idx=0)])
        
        # Final normalization
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language modeling head (shared with base model in practice)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def post_init(self):
        """Initialize weights."""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize module weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        base_model_hidden_states: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Forward pass of Eagle3 model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for caching
            inputs_embeds: Pre-computed input embeddings
            base_model_hidden_states: Base model hidden states (required for Eagle3)
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dictionary
            cache_position: Cache position
            
        Returns:
            Model outputs with logits and hidden states
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Get base model hidden states (required for Eagle3)
        if base_model_hidden_states is None:
            raise ValueError("base_model_hidden_states is required for Eagle3 model")
        
        # Prepare position embeddings
        batch_size, seq_length, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)
        print("################base_model_hidden_states ", base_model_hidden_states.shape)
        # Rotary embeddings
        hidden_states = torch.cat([inputs_embeds, base_model_hidden_states], dim=-1)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        
        # Initialize past key values
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0].shape[2]
        
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                device=inputs_embeds.device,
            )
        
        # Forward through decoder layer
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Forward through Eagle3 layer
            layer_outputs = layer(
                input_embeds=inputs_embeds,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            
            if output_attentions:
                all_self_attns += (layer_outputs[1] if len(layer_outputs) > 1 else None,)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            return tuple(
                v
                for v in [logits, all_hidden_states, next_decoder_cache, all_self_attns]
                if v is not None
            )
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Export for backward compatibility
__all__ = ["Qwen2ForCausalLMEagle3", "Qwen2EagleDecoderLayer"]
