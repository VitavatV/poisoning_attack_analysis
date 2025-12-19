# Corrected Experiment Mapping Summary

## Mapping Applied

NEW exp1 <- OLD exp0 (Baseline / Width-Depth Grid)
NEW exp2 <- OLD exp3 (Attack Type Comparison)
NEW exp3 <- OLD exp4 (IID vs Non-IID)  
NEW exp4 <- OLD exp1 (Width vs Depth Grid)
NEW exp5 <- OLD exp2 (Batch Size / Data Ordering)

## Migration Results

### NEW Exp1 (from OLD Exp0)
- MNIST: 48 rows
- CIFAR-10: 48 rows
- Purpose: Baseline experiments, width-depth grid
- Total: 96 rows

### NEW Exp2 (from OLD Exp3)
- MNIST: 27 rows
- CIFAR-10: 27 rows
- Purpose: Attack type comparison (label_flip vs random_noise)
- Total: 54 rows

### NEW Exp3 (from OLD Exp4)
- MNIST: 6 rows  
- CIFAR-10: 6 rows
- Purpose: Data heterogeneity (IID vs Non-IID analysis)
- Total: 12 rows

### NEW Exp4 (from OLD Exp1)
- MNIST: 20 rows
- CIFAR-10: 20 rows
- Purpose: Width and depth scaling analysis
- Total: 40 rows

### NEW Exp5 (from OLD Exp2)
- MNIST: 30 rows
- CIFAR-10: 30 rows
- Purpose: Mechanism analysis (batch size, data ordering)
- Total: 60 rows

## Total Migrated

**262 rows** across all experiments

## Coverage by New Config

### Exp1 (config_exp1.yaml)
- Required: 5 seeds × 2 models × 2 datasets × 4 widths × 4 depths × 3 poison = 960
- Migrated: 96 (10%)
- Gap: 864

### Exp2 (config_exp2.yaml)
- Required: 5 seeds × 2 models × 2 datasets × 3 batch × 3 ordering × 3 poison = 540
- Migrated: 60 (11%)
- Gap: 480

### Exp3 (config_exp3.yaml)
- Required: 5 seeds × 2 models × 2 datasets × 2 attacks × 3 poison = 120
- Migrated: 54 (45%)
- Gap: 66

### Exp4 (config_exp4.yaml)
- Required: 5 seeds × 2 models × 2 datasets × 4 alphas × 3 poison = 240
- Migrated: 12 (5%)
- Gap: 228

### Exp5 (config_exp5.yaml)
- Required: 5 seeds × 2 models × 2 datasets × 2 aggregators × 3 poison = 120
- Migrated: 60 (50%)
- Gap: 60

## Total Required vs Migrated

- Total required: 1,980 experiments
- Total migrated: 262 (13%)
- Total gap: 1,718 (87%)

## What Was Migrated

All migrated data represents:
- Model type: CNN only
- Seed: 42 (single seed)

Still needed for each config:
- Logistic Regression experiments
- Additional seeds (101, 2024, 3141, 9876)

## Files Created

Results are now in:
- results_exp1_mnist/final_results.csv (96 rows)
- results_exp1_cifar10/final_results.csv
- results_exp2_mnist/final_results.csv  
- results_exp2_cifar10/final_results.csv
- results_exp3_mnist/final_results.csv
- results_exp3_cifar10/final_results.csv
- results_exp4_mnist/final_results.csv
- results_exp4_cifar10/final_results.csv
- results_exp5_mnist/final_results.csv
- results_exp5_cifar10/final_results.csv

## Next Steps

1. Review migrated files to ensure correctness
2. Start experiment manager (will load these 262 completed experiments)
3. Workers will run remaining 1,718 new experiments

The manager will automatically skip the 262 migrated experiments.
