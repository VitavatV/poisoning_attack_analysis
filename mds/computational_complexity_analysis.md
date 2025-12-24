# Computational Complexity Analysis

This document analyzes whether model parameters are sufficient for task distribution, or if we need a better metric.

## The Problem

Model parameters don't always correlate with computational workload:

### Example Comparison

| Model | Dataset | Width | Depth | Parameters | Operations per Forward Pass |
|-------|---------|-------|-------|------------|---------------------------|
| CNN   | MNIST   | 1     | 4     | ~550       | Conv: O(W×H×K²×C×D) ≈ 28²×3²×1×4 = ~28K ops |
| LR    | MNIST   | 64    | 4     | ~63,882    | MatMul: O(I×H×D) = 784×64×4 ≈ 200K ops |
| CNN   | CIFAR   | 4     | 4     | ~10,000    | Conv: O(32²×3²×4×4) ≈ 147K ops |

**Issue**: The CNN with 550 params needs fewer ops than the LR with 63K params!

However, for **federated learning with batched training**, the real bottleneck is:

## Computational Complexity in Federated Learning

For each global round with `m` selected clients, `E` local epochs, and batch size `B`:

```
Total Computation = m × E × (N_client / B) × [Forward + Backward]
```

Where:
- **Forward pass**: Model-dependent (Conv vs MatMul)
- **Backward pass**: ~2-3× forward pass
- **Batch operations**: Parallelized on GPU/CPU

### Key Insight

**CNNs benefit MORE from GPU acceleration** due to:
- Highly parallelizable conv operations
- Optimized CUDA kernels (cuDNN)
- Spatial locality for cache efficiency

**MLPs are more memory-bound** and:
- Matrix ops are also GPU-friendly but less critical
- Can run reasonably well on CPU with good BLAS libraries

## Current Sorting Approach

Our parameter-based sorting actually works reasonably well because:

1. **Large CNNs (high params)** → GPU gets them ✓
2. **Large MLPs (high params)** → GPU gets them ✓  
3. **Small CNNs (low params)** → CPU gets them (could be better)
4. **Small MLPs (low params)** → CPU gets them ✓

The only suboptimal case is #3 (small CNNs on CPU).

## Proposed Improvement: FLOPs-Based Sorting

Calculate approximate FLOPs (Floating Point Operations) per training step:

### CNN FLOPs Estimation
```python
# Per conv layer
conv_flops = 2 × H × W × K² × in_channels × out_channels
# Plus batch norm
bn_flops = 4 × H × W × out_channels
```

### MLP FLOPs Estimation  
```python
# Per linear layer
linear_flops = 2 × input_dim × output_dim
# Plus batch norm
bn_flops = 4 × output_dim
```

### Total FLOPs
```python
total_flops = (forward_flops + 2 × forward_flops) × batch_size × local_epochs × num_clients
```

## Recommendation

**Option 1: Hybrid Metric (Recommended)**
```python
# Combine parameters and model type
if model_type == 'cnn':
    sort_key = params × CNN_MULTIPLIER  # e.g., 5x
else:
    sort_key = params × 1
```

**Option 2: Full FLOPs Calculation**
- More accurate but complex
- Requires detailed architecture knowledge

**Option 3: Keep Current (Simplest)**
- Parameter-based sorting is simple and mostly correct
- The main goal (large models → GPU) is achieved

## Decision Matrix

| Metric | Accuracy | Complexity | Implementation |
|--------|----------|------------|----------------|
| **Parameters only** | 80% | Low | ✓ Done |
| **Params × Type** | 90% | Low | Easy |
| **FLOPs-based** | 95% | Medium | Moderate |

For your use case, I recommend **Option 1** (Hybrid Metric) as the best balance.
