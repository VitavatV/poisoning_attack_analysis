# Model Type Comparison: CNN vs Scalable MLP

## Summary

Successfully added model type comparison functionality to the experiment framework. Both CNN and MLP (Logistic Regression) now use identical `width_factor` and `depth` parameters for fair comparison.

---

## Files Modified

### 1. [models.py](file:///c:/github/poisoning_attack_analysis/models.py)
- **Updated `LogisticRegression` class** → Scalable Multi-Layer Perceptron
- Added `width_factor` and `depth` parameters
- Architecture mirrors CNN scaling behavior

### 2. [experiment_runner_gpu.py](file:///c:/github/poisoning_attack_analysis/experiment_runner_gpu.py)
- Added `create_model()` helper function
- Updated 2 model instantiation locations
- Added `model_type` to CSV results

### 3. [experiment_manager.py](file:///c:/github/poisoning_attack_analysis/experiment_manager.py)
- Added `model_type` to task signatures
- Ensures proper task tracking and deduplication

### 4. All Config Files
- Added `model_type: ["cnn", "lr"]` to combinations
- Doubled experiment count (both models tested per config)

---

## Model Architectures

### CNN (Convolutional Neural Network)
```
Input (H×W×C) → [Conv(W) → BN → ReLU → MaxPool] × D → Flatten → Linear → Output
```
- Spatial feature extraction
- Parameter sharing through convolutions
- Efficient for image data

### MLP (Multi-Layer Perceptron / Logistic Regression)
```
Input → Flatten → [Linear(64×W) → ReLU → BN] × (D-1) → Linear → Output
```
- Fully-connected architecture
- No spatial inductive bias
- Baseline for comparison

**Special Case (depth=1):**
```
Input → Flatten → Linear → Output  (Pure Logistic Regression)
```

---

## Parameter Scaling

Both models scale identically with `width_factor` and `depth`:

| Config | CNN Params (MNIST) | MLP Params (MNIST) | Ratio |
|--------|-------------------|-------------------|-------|
| W=1, D=1 | 7,862 | 7,850 | 1.0x |
| W=1, D=4 | 548 | 59,594 | 109x |
| W=4, D=4 | 2,486 | 336,650 | 135x |
| W=16, D=16 | 35,642 | 15,539,210 | 436x |

**Key Insight:** MLP has significantly more parameters than CNN for same width/depth due to fully-connected layers.

---

## Experiment Impact

### Before
- 4,260 experiments (5 experiments × ~852 configs)
- Only CNN tested

### After
- **8,520 experiments** (doubled)
- Both CNN and MLP tested for each configuration
- Direct comparison of architectures under same conditions

### Estimated Runtime
- **With 3 GPUs:** ~12-20 days (was 6-10 days)
- Each experiment runs twice (CNN + MLP)

---

## Research Questions Enabled

This comparison allows investigating:

1. **Architectural Inductive Bias**
   - Does CNN's spatial structure improve robustness?
   - Do fully-connected MLPs behave differently under poisoning?

2. **Parameter Efficiency**
   - Can smaller CNNs match larger MLPs?
   - How does parameter count affect robustness?

3. **Double Descent Phenomenon**
   - Does double descent occur in both architectures?
   - Are the critical thresholds different?

4. **Attack Transferability**
   - Do attacks affect both architectures similarly?
   - Is one architecture inherently more robust?

5. **Defense Mechanisms**
   - Do defenses (e.g., median aggregation) work equally for both?
   - Architecture-specific vulnerabilities?

---

## Usage

### Running Experiments

No changes needed - existing workflow works:

```bash
# Terminal 1: Start manager
python experiment_manager.py

# Terminal 2+: Start workers
python experiment_runner_gpu.py
```

### Results Format

CSV now includes `model_type` column:

| phase | dataset | model_type | width_factor | depth | ... |
|-------|---------|------------|--------------|-------|-----|
| exp1 | mnist | cnn | 4 | 4 | ... |
| exp1 | mnist | lr | 4 | 4 | ... |
| exp1 | cifar10 | cnn | 16 | 16 | ... |
| exp1 | cifar10 | lr | 16 | 16 | ... |

### Analysis

Compare CNN vs MLP:

```python
import pandas as pd

df = pd.read_csv('results_exp1_mnist/final_results.csv')

# Filter by width=4, depth=4
subset = df[(df['width_factor']==4) & (df['depth']==4)]

# Compare CNN vs MLP accuracy
cnn_acc = subset[subset['model_type']=='cnn']['mean_test_acc'].mean()
mlp_acc = subset[subset['model_type']=='lr']['mean_test_acc'].mean()

print(f"CNN: {cnn_acc:.4f}, MLP: {mlp_acc:.4f}")
```

---

## Verification

✅ **Test script passed**: [test_scalable_models.py](file:///c:/github/poisoning_attack_analysis/test_scalable_models.py)

Confirmed:
- Both models forward pass successfully
- Correct output shapes
- Parameter counts scale as expected
- Works with MNIST (28×28, 1ch) and CIFAR-10 (32×32, 3ch)

---

## IEEE Publication Impact

This addition strengthens the paper by:

✅ **Broader Generalization**: Findings apply beyond CNNs  
✅ **Architectural Comparison**: Isolates effect of model structure  
✅ **Baseline Establishment**: MLP/LR as classical ML baseline  
✅ **Comprehensive Analysis**: More complete experimental coverage

---

## Backward Compatibility

✅ **Existing results preserved**: Task signatures include model_type  
✅ **Default behavior**: Config without model_type defaults to 'cnn'  
✅ **Old configs work**: Manager recognizes and skips completed CNN experiments

---

## Next Steps

1. ✅ Models implemented and tested
2. ✅ Configs updated
3. ✅ Runner and manager updated
4. 🔄 **Ready to run experiments**
5. 📊 **Analyze results comparing CNN vs MLP**
6. 📝 **Write paper sections on architectural comparison**
