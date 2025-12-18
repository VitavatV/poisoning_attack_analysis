"""
Figure Utility Module for IEEE Publication Format
Generates figures at 600 DPI in both 3.5" and 7.16" widths
"""
import matplotlib.pyplot as plt
import os
from pathlib import Path


def save_ieee_figure(figure_name, exp_name, dataset, figure_num, fig=None):
    """
    Save figure in IEEE publication format with two sizes.
    
    Args:
        figure_name: Base name for the figure (e.g., 'heatmap_width_vs_depth_clean')
        exp_name: Experiment name (e.g., 'exp0')
        dataset: Dataset name (e.g., 'cifar10', 'mnist')
        figure_num: Figure number as string (e.g., '01', '02')
        fig: matplotlib figure object (if None, uses current figure)
    
    Example:
        save_ieee_figure('heatmap_clean', 'exp0', 'cifar10', '01')
        # Saves: figure/exp0/exp0_cifar10_01_3p5inch.png
        #        figure/exp0/exp0_cifar10_01_7p16inch.png
    """
    if fig is None:
        fig = plt.gcf()
    
    # Create output directory
    output_dir = Path('figure') / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get current figure size to calculate aspect ratio
    current_size = fig.get_size_inches()
    aspect_ratio = current_size[1] / current_size[0]
    
    # IEEE column widths in inches
    widths = {
        '3p5inch': 3.5,
        '7p16inch': 7.16
    }
    
    saved_files = []
    for width_name, width_value in widths.items():
        # Calculate height based on aspect ratio
        height = width_value * aspect_ratio
        
        # Set figure size
        fig.set_size_inches(width_value, height)
        
        # Generate filename
        filename = f"{exp_name}_{dataset}_{figure_num}_{width_name}.png"
        filepath = output_dir / filename
        
        # Save with 600 DPI
        fig.savefig(filepath, dpi=600, bbox_inches='tight')
        saved_files.append(str(filepath))
    
    # Reset to original size
    fig.set_size_inches(current_size)
    
    print(f"✓ Saved IEEE figures: {exp_name}_{dataset}_{figure_num} (3.5\" & 7.16\")")
    
    return saved_files
