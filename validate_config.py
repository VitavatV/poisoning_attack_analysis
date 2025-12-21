"""
Configuration Validation Utility

Validates experiment configurations to detect potentially unstable model architectures
and parameter combinations before running expensive experiments.
"""

import logging
import sys
from typing import Dict, List, Tuple


def validate_model_architecture(config: Dict) -> List[str]:
    """
    Validate model architecture parameters for potential training instability.
    
    Returns:
        List of warning messages (empty if no issues detected)
    """
    warnings = []
    
    model_type = config.get('model_type', 'cnn')
    depth = config.get('depth', 4)
    width_factor = config.get('width_factor', 4)
    
    # Check for extremely deep networks
    if depth > 64:
        warnings.append(
            f"⚠️ CRITICAL: Extremely deep {model_type.upper()} (depth={depth}). "
            f"Very high risk of training instability! Consider depth <= 64."
        )
    
    # Model-specific validation
    if model_type == 'lr':
        # Check depth-to-width ratio for LR models
        depth_width_ratio = depth / max(width_factor, 1)
        
        if depth_width_ratio > 8:
            warnings.append(
                f"⚠️ WARNING: LogisticRegression too deep and narrow "
                f"(depth={depth}, width={width_factor}, ratio={depth_width_ratio:.1f}). "
                f"May cause gradient instability. "
                f"Recommended: depth <= {width_factor * 8} OR width >= {depth // 8}"
            )
        
        if depth > 32 and width_factor < 8:
            warnings.append(
                f"⚠️ WARNING: Very deep LR model with narrow layers "
                f"(depth={depth}, width={width_factor}). "
                f"Gradient clipping and reduced LR are strongly recommended."
            )
    
    elif model_type == 'cnn':
        # Check for very narrow CNNs
        if width_factor < 1:
            warnings.append(
                f"⚠️ WARNING: Very narrow CNN (width_factor={width_factor}). "
                f"May limit learning capacity. Consider width_factor >= 1."
            )
        
        # Info for very deep CNNs (less critical than LR)
        if depth > 32:
            warnings.append(
                f"ℹ️ INFO: Deep CNN (depth={depth}). Training time may be longer, "
                f"but CNNs are generally stable at this depth."
            )
    
    return warnings


def validate_training_params(config: Dict) -> List[str]:
    """
    Validate training hyperparameters for potential issues.
    
    Returns:
        List of warning messages (empty if no issues detected)
    """
    warnings = []
    
    lr = config.get('lr', 0.01)
    depth = config.get('depth', 4)
    model_type = config.get('model_type', 'cnn')
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    # Check LR for deep networks
    if model_type == 'lr' and depth > 32:
        if lr > 0.01:
            warnings.append(
                f"⚠️ WARNING: High learning rate ({lr}) for deep LR model (depth={depth}). "
                f"Recommended: lr <= 0.01 for depth > 32"
            )
    
    # Check gradient clipping
    if max_grad_norm is None and depth > 16:
        warnings.append(
            f"⚠️ WARNING: Gradient clipping disabled for deep network (depth={depth}). "
            f"Recommended: max_grad_norm=1.0 for stability"
        )
    
    if max_grad_norm is not None and max_grad_norm > 5.0:
        warnings.append(
            f"⚠️ INFO: Large gradient clipping threshold ({max_grad_norm}). "
            f"Consider max_grad_norm <= 2.0 for better stability"
        )
    
    return warnings


def validate_config(config: Dict, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Complete configuration validation.
    
    Args:
        config: Experiment configuration dictionary
        verbose: If True, print warnings to console
    
    Returns:
        (is_valid, warnings): Boolean indicating if config is safe to run,
                              and list of warning messages
    """
    all_warnings = []
    
    # Validate architecture
    arch_warnings = validate_model_architecture(config)
    all_warnings.extend(arch_warnings)
    
    # Validate training parameters
    train_warnings = validate_training_params(config)
    all_warnings.extend(train_warnings)
    
    # Determine if configuration is safe
    critical_issues = [w for w in all_warnings if 'CRITICAL' in w]
    is_valid = len(critical_issues) == 0
    
    if verbose:
        if all_warnings:
            print("\n" + "="*70)
            print("⚠️  CONFIGURATION VALIDATION WARNINGS")
            print("="*70)
            for warning in all_warnings:
                print(f"\n{warning}")
            print("\n" + "="*70)
            
            if not is_valid:
                print("\n❌ CRITICAL ISSUES DETECTED - Configuration may cause training failure")
            else:
                print("\n⚠️  Warnings detected - Proceed with caution")
        else:
            print("\n✅ Configuration validation passed - No issues detected")
    
    return is_valid, all_warnings


def suggest_fixes(config: Dict) -> Dict:
    """
    Suggest improved configuration parameters based on validation.
    
    Returns:
        Dictionary with suggested parameter changes
    """
    suggestions = {}
    
    model_type = config.get('model_type', 'cnn')
    depth = config.get('depth', 4)
    width_factor = config.get('width_factor', 4)
    lr = config.get('lr', 0.01)
    
    # Suggest LR reduction for deep networks
    if model_type == 'lr' and depth > 32:
        import numpy as np
        lr_scale = 1.0 / np.sqrt(depth / 32.0)
        suggested_lr = lr * lr_scale
        suggestions['lr'] = suggested_lr
        suggestions['lr_reason'] = f"Scaled down for depth={depth} (scale factor: {lr_scale:.3f})"
    
    # Suggest gradient clipping if not set
    if config.get('max_grad_norm') is None and depth > 16:
        suggestions['max_grad_norm'] = 1.0
        suggestions['max_grad_norm_reason'] = "Added for deep network stability"
    
    # Suggest width increase for very narrow deep networks
    if model_type == 'lr' and depth > 32 and width_factor < 8:
        suggested_width = max(8, depth // 8)
        suggestions['width_factor'] = suggested_width
        suggestions['width_factor_reason'] = f"Increased to improve depth/width ratio"
    
    return suggestions


if __name__ == "__main__":
    # Example usage
    test_config = {
        'dataset': 'mnist',
        'model_type': 'lr',
        'width_factor': 4,
        'depth': 64,
        'poison_ratio': 0.3,
        'lr': 0.02,
        'global_rounds': 3000,
        'local_epochs': 5,
        'batch_size': 128,
        'num_clients': 10,
        'max_grad_norm': 1.0
    }
    
    print("Testing Configuration Validator")
    print("="*70)
    print(f"Config: model_type={test_config['model_type']}, depth={test_config['depth']}, "
          f"width={test_config['width_factor']}, lr={test_config['lr']}")
    
    is_valid, warnings = validate_config(test_config, verbose=True)
    
    if not is_valid or warnings:
        print("\n" + "="*70)
        print("💡 SUGGESTED FIXES")
        print("="*70)
        suggestions = suggest_fixes(test_config)
        
        if suggestions:
            for key, value in suggestions.items():
                if not key.endswith('_reason'):
                    reason = suggestions.get(f"{key}_reason", "")
                    print(f"\n{key}: {test_config.get(key, 'N/A')} -> {value}")
                    if reason:
                        print(f"   Reason: {reason}")
        else:
            print("\nNo automatic fixes available. Manual review recommended.")
