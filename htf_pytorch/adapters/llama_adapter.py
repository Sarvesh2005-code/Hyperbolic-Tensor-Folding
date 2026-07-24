import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

from htf_pytorch.attention import HyperbolicAttention
from htf_pytorch.manifold import expmap0, logmap0

class HyperbolicLlamaAdapter(nn.Module):
    """
    Adapter that intercepts the Llama Architecture (used by Llama-2, Llama-3, TinyLlama).
    It extracts the Rotary Positional Embeddings (RoPE), applies them in Euclidean space,
    and then maps the tensors to the Poincare ball for Hyperbolic Tensor Folding.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        # We initialize the original attention just to hijack its Q/K/V projections and RoPE logic
        self.orig_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.htf_attn = HyperbolicAttention(c=1.0)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()
        
        # 1. Extract standard Euclidean projections
        query_states = self.orig_attn.q_proj(hidden_states)
        key_states = self.orig_attn.k_proj(hidden_states)
        value_states = self.orig_attn.v_proj(hidden_states)
        
        # Reshape for RoPE
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply Rotary Positional Embeddings (RoPE)
        if position_ids is not None and hasattr(self.orig_attn, "rotary_emb"):
            cos, sin = self.orig_attn.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
        # Broadcast Grouped-Query Attention (GQA) K and V to match Q
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Flatten back out for HTF processing [Batch, Seq, Dim]
        query_states = query_states.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        key_states = key_states.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        value_states = value_states.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        
        # 2. Map standard Euclidean embeddings into the non-Euclidean Poincare ball
        q_hyp = expmap0(query_states)
        k_hyp = expmap0(key_states)
        v_hyp = expmap0(value_states)
        
        # 3. Run Hyperbolic Tensor Folding (HTF) Attention
        out_hyp, attn_weights = self.htf_attn(q_hyp, k_hyp, v_hyp)
        
        # 4. Map back to Euclidean space
        out_eucl = logmap0(out_hyp)
        
        # Final output projection
        attn_output = self.orig_attn.o_proj(out_eucl)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value,)
            
        return outputs


def inject_htf_llama(model, config=None):
    """
    Iterates through a Llama-architecture model and surgically replaces 
    LlamaAttention with HyperbolicLlamaAdapter.
    """
    if config is None:
        config = getattr(model, "config", None)
    
    for name, module in model.named_children():
        if module.__class__.__name__ == "LlamaAttention":
            # Pass the layer index if it exists to maintain RoPE caching
            layer_idx = getattr(module, "layer_idx", None)
            setattr(model, name, HyperbolicLlamaAdapter(config, layer_idx=layer_idx))
        else:
            inject_htf_llama(module, config)
