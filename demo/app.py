import os
import sys
# Add root of project to path so we can import from htf_pytorch and examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import plotly.graph_objects as go
from transformers import GPT2Model, GPT2Config
import tracemalloc
import time

from examples.hf_integration import inject_htf

st.set_page_config(page_title="HTF Memory Benchmark", page_icon="🚀", layout="wide")

st.title("Hyperbolic Tensor Folding (HTF) 🚀")
st.markdown("### The Non-Euclidean Attention Memory Benchmark")

st.markdown("""
Standard Euclidean Attention forces the KV-Cache to grow polynomially. If you want a massive context window, you need massive VRAM.

**HTF** maps attention into the Poincaré disk. Because hyperbolic space expands exponentially towards its boundary, we fold infinite hierarchical semantic trees into a finite coordinate radius—drastically shrinking RAM requirements.
""")

with st.sidebar:
    st.header("Benchmark Settings")
    max_seq_len = st.slider("Max Sequence Length (Tokens)", min_value=256, max_value=8192, value=2048, step=256)
    num_points = st.slider("Data Points (Resolution)", min_value=5, max_value=20, value=10)
    run_btn = st.button("Run Live Benchmark", type="primary")

if run_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    config = GPT2Config(n_layer=4, n_head=4, n_embd=128)
    std_model = GPT2Model(config)
    
    htf_model = GPT2Model(config)
    inject_htf(htf_model)

    # Generate sequence lengths to test
    step_size = max(128, max_seq_len // num_points)
    seq_lengths = list(range(128, max_seq_len + 1, step_size))
    
    std_mems = []
    htf_mems = []

    for i, sl in enumerate(seq_lengths):
        status_text.text(f"Benchmarking Sequence Length: {sl} tokens...")
        dummy_input = torch.randint(0, config.vocab_size, (1, sl))
        
        # Standard
        tracemalloc.start()
        _ = std_model(dummy_input)
        _, peak_std = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        std_mems.append(peak_std / (1024**2))
        
        # HTF
        tracemalloc.start()
        _ = htf_model(dummy_input)
        _, peak_htf = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        htf_mems.append(peak_htf / (1024**2))
        
        progress_bar.progress((i + 1) / len(seq_lengths))
        time.sleep(0.1) # UI refresh

    status_text.text("Benchmark Complete!")

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seq_lengths, y=std_mems, mode='lines+markers', name='Standard Attention', line=dict(color='#ff4b4b', width=3)))
    fig.add_trace(go.Scatter(x=seq_lengths, y=htf_mems, mode='lines+markers', name='HTF Attention', line=dict(color='#00cc96', width=3)))
    
    fig.update_layout(
        title="Peak RAM Usage vs Context Length",
        xaxis_title="Context Length (Tokens)",
        yaxis_title="Peak RAM (MB)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    savings = (1 - (htf_mems[-1] / std_mems[-1])) * 100 if std_mems[-1] > 0 else 0
    st.success(f"**Result:** At {seq_lengths[-1]} tokens, HTF saved **{savings:.2f}%** of peak RAM compared to standard PyTorch Attention!")
