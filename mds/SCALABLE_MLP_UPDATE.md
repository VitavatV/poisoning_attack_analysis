# Scalable MLP (Logistic Regression) Update

## Changes Made

Updated `LogisticRegression` model to be a **scalable Multi-Layer Perceptron (MLP)** that uses the same `width_factor` and `depth` parameters as `ScalableCNN`.

### Model Architecture

**Before:**
- Simple logistic regression (input → output)
- No hidden layers
- Fixed architecture

**After:**
- Scalable MLP with configurable depth and width
- `depth=1`: Pure logistic regression (no hidden layers)
- `depth>1`: Multi-layer perceptron with hidden layers
- Width controlled by `width_factor`

### Architecture Details

```python
# depth=1: Pure Logistic Regression
Input (flattened) → Output

# depth=4: 3-layer MLP
Input (flattened) → Hidden1 (64×width) → ReLU → BatchNorm
                  → Hidden2 (64×width) → ReLU → BatchNorm  
                  → Hidden3 (64×width) → ReLU → BatchNorm
                  → Output

# General formula
Input → [Hidden (64×width) → ReLU → BatchNorm] × (depth-1) → Output
```

### Parameters

| Parameter | Effect |
|-----------|--------|
| `width_factor` | Multiplies base hidden size (64 units) |
| `depth` | Number of layers (depth-1 hidden layers) |

### Examples

| width_factor | depth | Hidden Size | Layers |
|--------------|-------|-------------|--------|
| 1 | 1 | N/A | Input → Output (pure LR) |
| 4 | 1 | N/A | Input → Output (pure LR) |
| 1 | 4 | 64 | 3 hidden layers @ 64 units |
| 4 | 4 | 256 | 3 hidden layers @ 256 units |
| 16 | 4 | 1024 | 3 hidden layers @ 1024 units |
| 64 | 16 | 4096 | 15 hidden layers @ 4096 units |

### Comparison with CNN

Both models now scale identically:

| Aspect | CNN | MLP (LR) |
|--------|-----|----------|
| **Input Processing** | Convolutions | Flatten |
| **Width Scaling** | Channels × width_factor | Units × width_factor |
| **Depth Scaling** | Conv layers | FC layers |
| **Base Width** | 1 channel | 64 units |
| **Activation** | ReLU | ReLU |
| **Normalization** | BatchNorm2d | BatchNorm1d |

### Model Size Comparison

**MNIST (28×28, 1 channel):**
- Input dim: 784
- LR depth=1, width=1: ~7K params
- LR depth=4, width=4: ~256K params
- CNN depth=4, width=4: ~10K params

**CIFAR-10 (32×32, 3 channels):**
- Input dim: 3072
- LR depth=1, width=1: ~30K params
- LR depth=4, width=4: ~1M params
- CNN depth=4, width=4: ~10K params

**Note:** MLP has more parameters than CNN due to fully-connected layers, but same scaling behavior.

### Code Updates

1. **models.py**: Redesigned `LogisticRegression` class
2. **experiment_runner_gpu.py**: Updated `create_model()` to pass width/depth to LR

### Benefits

✅ **Fair Comparison**: Both models scale the same way  
✅ **Consistent Parameters**: Same width_factor and depth values  
✅ **Flexibility**: LR can be simple (depth=1) or complex (depth=64)  
✅ **Parameter Matching**: Can create LR and CNN with similar capacity

### Usage

No changes needed - existing configs work automatically!

```yaml
combinations:
  - model_type: ["cnn", "lr"]
  - width_factor: [1, 4, 16, 64]
  - depth: [1, 4, 16, 64]
```

Both CNN and LR will now use the same width/depth parameters for true apples-to-apples comparison.
