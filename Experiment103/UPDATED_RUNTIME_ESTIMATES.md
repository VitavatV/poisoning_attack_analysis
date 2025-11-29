# Updated Runtime Estimates (CPU vs RTX 4060)

Based on the updated configurations (Exp 0-4), here are the estimated runtimes.

## Experiment Counts

| Experiment | Configuration Changes | Total Combinations |
| :--- | :--- | :--- |
| **EXP 0** | Width(6) × Depth(6) × Poison(3) | **108** |
| **EXP 1** | Width(1) × Poison(3) × Alpha(4) | **12** |
| **EXP 2** | Agg(2) × Poison(3) × Alpha(2) | **12** |
| **EXP 3** | Batch(3) × Order(3) × Poison(3) × Alpha(2) | **54** |
| **EXP 4** | Type(2) × Poison(3) × Alpha(2) | **12** |
| **TOTAL** | | **198 Experiments** |

## Runtime Estimates

**Assumptions:**
*   **RTX 4060:** ~20-25 minutes per experiment (average).
    *   EXP 0 (Variable sizes): ~25 mins/exp.
    *   EXP 1-4 (Fixed W=16, D=4): ~20 mins/exp.
*   **CPU:** ~15x slower than RTX 4060 (conservative estimate for CNN training).
    *   ~5-6 hours per experiment.

### Breakdown

| Experiment | RTX 4060 Estimate | CPU Estimate |
| :--- | :--- | :--- |
| **EXP 0** | 108 × 25m = **~45 hours** (1.9 days) | 108 × 6.25h = **~675 hours** (28 days) |
| **EXP 1** | 12 × 20m = **~4 hours** | 12 × 5h = **~60 hours** (2.5 days) |
| **EXP 2** | 12 × 20m = **~4 hours** | 12 × 5h = **~60 hours** (2.5 days) |
| **EXP 3** | 54 × 20m = **~18 hours** | 54 × 5h = **~270 hours** (11 days) |
| **EXP 4** | 12 × 20m = **~4 hours** | 12 × 5h = **~60 hours** (2.5 days) |
| **TOTAL** | **~75 hours (3.1 days)** | **~1125 hours (47 days)** |

## Recommendation

*   **RTX 4060:** Highly recommended. The entire suite can be finished in about **3 days** running 24/7.
*   **CPU:** Not feasible for the full suite.
    *   If you must use CPU, limit to **EXP 1 or EXP 2 only** (would take ~2-3 days each).
    *   EXP 0 and EXP 3 are impossible on CPU within a reasonable timeframe.

## Memory Note (RTX 4060 8GB)
*   **EXP 0:** Contains `width=32, depth=32`. This is the largest model.
    *   Previous tests suggested `width=64` was the limit. `width=32` should fit, but monitor VRAM.
    *   If OOM occurs, reduce `batch_size` in `config_exp0.yaml` for those specific runs or globally.
