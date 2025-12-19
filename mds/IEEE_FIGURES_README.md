# Quick Reference: IEEE Figure Generation

## Generated Files Summary

All notebooks now save figures in the `figure/` directory with this naming pattern:
```
figure/{exp_name}/{exp_name}_{dataset}_{number}_{width}.png
```

Example: `figure/exp0/exp0_cifar10_01_3p5inch.png`

## File Naming Convention

- **exp_name**: Experiment identifier (exp0, exp1, exp2, exp3, exp4)
- **dataset**: Dataset name (cifar10, mnist)
- **number**: Figure number (01, 02, 03, ...)
- **width**: Column width (3p5inch or 7p16inch)

## Quick Stats

| Experiment | Figures per Dataset | Total Files |
|------------|---------------------|-------------|
| exp0       | 5                   | 20          |
| exp1       | 3                   | 12          |
| exp2       | 3                   | 12          |
| exp3       | 3                   | 12          |
| exp4       | 3                   | 12          |
| **TOTAL**  | **17**              | **68**      |

_Note: Each figure generates 2 files (3.5" and 7.16" widths) × 2 datasets_

## Figure Specifications

- **DPI**: 600 (publication quality)
- **Format**: PNG
- **Widths**: 3.5" (single column) and 7.16" (double column)
- **Heights**: Automatically calculated to maintain aspect ratio

##To Generate Figures

1. Open Jupyter Notebook
2. Run any `analysis_exp*.ipynb` notebook
3. Figures automatically save to `figure/` directory
4. Each figure generates both 3.5" and 7.16" versions

## Example LaTeX Usage

**Single Column**:
```latex
\includegraphics[width=\columnwidth]{figure/exp0/exp0_cifar10_01_3p5inch.png}
```

**Double Column**:
```latex
\includegraphics[width=\textwidth]{figure/exp0/exp0_cifar10_01_7p16inch.png}
```

## Files Created

1. `figure_utils.py` - Core utility module
2. `modify_notebooks.py` - Automated modification script  
3. `test_ieee_figures.py` - Verification script
4. Modified all 10 `analysis_exp*.ipynb` notebooks

## Need Help?

- Check `walkthrough.md` for detailed documentation
- Run `python test_ieee_figures.py` to verify setup
- Review `implementation_plan.md` for technical details
