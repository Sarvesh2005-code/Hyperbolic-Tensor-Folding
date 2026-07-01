import torch
import torch.nn as nn
from .attention import HyperbolicMultiHeadAttention
from .manifold import expmap0, logmap0

class HuggingFaceHyperbolicAttentionWrapper(nn.Module):
    """
    Wraps our HyperbolicMultiHeadAttention to have an identical signature
    to a HuggingFace LlamaAttention module.
    """
    def __init__(self, config, c=1.0):
        super().__init__()
        self.config = config
        self.c = c
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        # We need projections to map Euclidean hidden states to Q, K, V
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.hyperbolic_attn = HyperbolicMultiHeadAttention(
            num_heads=self.num_heads, 
            embed_dim=self.hidden_size, 
            c=self.c
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # 1. Linear Projections (in Euclidean Space)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 2. Map to Poincare Ball
        q_poincare = expmap0(query_states, self.c)
        k_poincare = expmap0(key_states, self.c)
        v_poincare = expmap0(value_states, self.c)
        
        # (Note: KV-Cache and rotary embeddings handling omitted for brevity in this wrapper)
        
        # 3. Hyperbolic Attention
        attn_output_poincare, attn_weights = self.hyperbolic_attn(q_poincare, k_poincare, v_poincare)
        
        # 4. Map back to Euclidean
        attn_output = logmap0(attn_output_poincare, self.c)
        
        # 5. Output Projection
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs

def swap_llama_attention(model, c=1.0):
    """
    Recursively swaps standard LlamaAttention modules with our Hyperbolic wrapper.
    """
    import transformers
    for name, module in model.named_children():
        # Match LlamaAttention loosely by class name to avoid importing specific version
        if module.__class__.__name__ == "LlamaAttention":
            print(f"Swapping {name} to HyperbolicAttentionWrapper...")
            hyperbolic_attn = HuggingFaceHyperbolicAttentionWrapper(model.config, c=c)
            # Copy weights for projections if possible (optional)
            hyperbolic_attn.q_proj.weight = module.q_proj.weight
            hyperbolic_attn.k_proj.weight = module.k_proj.weight
            hyperbolic_attn.v_proj.weight = module.v_proj.weight
            hyperbolic_attn.o_proj.weight = module.o_proj.weight
            
            setattr(model, name, hyperbolic_attn)
        else:
            swap_llama_attention(module, c=c)
    return model
