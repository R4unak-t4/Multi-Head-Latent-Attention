# Multi-Head Latent Attention (MLA) Implementation

A PyTorch implementation of Multi-Head Latent Attention, an efficient attention mechanism that compresses key-value representations into a lower-dimensional latent space to reduce memory consumption during inference.

## Overview

Traditional multi-head attention mechanisms store full key-value pairs for each token, leading to significant memory overhead during inference. MLA addresses this by:

1. **Compressing KV pairs** into a lower-dimensional latent space
2. **Absorbing query projections** to reduce computational overhead
3. **Maintaining causal attention** for autoregressive generation


## Key Features

- **Memory Efficient**: Reduces KV cache size by projecting to latent dimensions
- **Absorbed Query**: Pre-computes query-key interactions for efficiency
- **Causal Masking**: Supports autoregressive text generation
- **Incremental Processing**: Supports efficient token-by-token generation

## Architecture Components

### 1. Latent Projection
```python
self.W_dkv = nn.Linear(d_model, kv_latent_dim, bias=False)  # Down projection
self.W_uk = nn.Linear(kv_latent_dim, d_model, bias=False)   # Up projection for keys
self.W_uv = nn.Linear(kv_latent_dim, d_model, bias=False)   # Up projection for values
```

### 2. Absorbed Query
The absorbed query pre-computes the interaction between query weights and up-projected key weights:
```python
absorbed = torch.matmul(self.W_q.weight, self.W_uk.weight)
self.absorbed_k = absorbed.view(self.n_heads, self.dh, -1)
```


### 3. Attention Computation
```python
# Compute attention scores using absorbed query
for h in range(self.n_heads):
    temp = torch.matmul(q[:,:,h], self.absorbed_k[h])
    attn_scores[:,h] = torch.bmm(temp, c_kv.transpose(1,2))
```

## Usage

### Basic Usage
```python
import torch
from mla import MLA

# Initialize model
model = MLA(d_model=512, n_heads=8, kv_latent_dim=256)

# Forward pass
x = torch.randn(1, 10, 512)  # (batch, seq_len, d_model)
output, kv_cache = model(x)
```

### Incremental Generation
```python
# Initial processing
x_initial = torch.randn(1, 5, 512)
output, cache = model(x_initial)

# Add new tokens incrementally
new_token = torch.randn(1, 1, 512)
output, cache = model(new_token, kv_cache=cache, past_length=cache.shape[1])
```

## Memory Comparison


### Standard Attention vs MLA
- **Standard KV Cache**: `batch_size × seq_len × n_heads × head_dim × 2 × 4 bytes`
- **MLA Latent Cache**: `batch_size × seq_len × kv_latent_dim × 4 bytes`

For typical configurations:
- Standard: ~20KB per 1000 tokens
- MLA: ~5KB per 1000 tokens (4x reduction)

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `d_model` | Model dimension | 512 |
| `n_heads` | Number of attention heads | 8 |
| `kv_latent_dim` | Latent KV dimension | 256 |

## Implementation Details

### Forward Pass Steps
1. **Latent Projection**: Project input to latent KV space
2. **Cache Management**: Concatenate with existing cache
3. **Value Computation**: Up-project latent values
4. **Attention Scores**: Compute using absorbed query
5. **Causal Masking**: Apply lower triangular mask
6. **Output Projection**: Combine attention heads

### Causal Masking
```python
mask = torch.tril(torch.ones(S, S_full, device=x.device), diagonal=past_length)
attn_scores = attn_scores.masked_fill(mask.view(1, 1, S, S_full) == 0, float('-inf'))
```

## Performance Benefits

1. **Memory Efficiency**: Significant reduction in KV cache size
2. **Computational Efficiency**: Absorbed query reduces matrix operations
3. **Scalability**: Better performance with longer sequences
4. **Compatibility**: Drop-in replacement for standard attention

## File Structure
```
mla/
├── mla.py              # Main implementation
├── README.md           # This file
├── examples/
│   ├── basic_usage.py
│   └── benchmarks.py
└── images/
    ├── mla_architecture.png
    ├── absorbed_query.png
    ├── memory_comparison.png
    └── causal_attention.png
```

## Requirements
- PyTorch >= 1.9.0
- Python >= 3.7

## Installation
```bash
pip install torch
```

## Examples

See the `examples/` directory for detailed usage examples and benchmarks.
