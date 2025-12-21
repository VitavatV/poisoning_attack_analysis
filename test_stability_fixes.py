"""
Quick test script to verify fixes for NaN/Inf training instability.
Tests the problematic configuration that was failing before.
"""

import torch
import numpy as np
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from models import LogisticRegression
from validate_config import validate_config, suggest_fixes

def test_model_initialization():
    """Test that LogisticRegression initializes properly with deep networks"""
    print("\n" + "="*70)
    print("TEST 1: Model Initialization")
    print("="*70)
    
    # Test problematic configuration: depth=64, width=4
    model = LogisticRegression(
        num_classes=10,
        width_factor=4,
        depth=64,
        in_channels=1,
        img_size=28
    )
    
    # Check for NaN/Inf in initial weights
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"❌ FAILED: NaN/Inf found in {name} after initialization")
            has_nan = True
    
    if not has_nan:
        print("✅ PASSED: No NaN/Inf in initial weights")
    
    # Check weight statistics
    for name, param in model.named_parameters():
        if 'weight' in name:
            mean = param.data.mean().item()
            std = param.data.std().item()
            print(f"   {name}: mean={mean:.6f}, std={std:.6f}")
    
    return not has_nan


def test_forward_pass():
    """Test that model can perform forward pass without NaN"""
    print("\n" + "="*70)
    print("TEST 2: Forward Pass")
    print("="*70)
    
    model = LogisticRegression(
        num_classes=10,
        width_factor=4,
        depth=64,
        in_channels=1,
        img_size=28
    )
    
    # Create dummy input
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    
    # Forward pass
    try:
        output = model(x)
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("❌ FAILED: NaN/Inf in output")
            return False
        
        print(f"✅ PASSED: Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        return True
    except Exception as e:
        print(f"❌ FAILED: Forward pass error: {e}")
        return False


def test_backward_pass():
    """Test that model can perform backward pass without NaN"""
    print("\n" + "="*70)
    print("TEST 3: Backward Pass (Single Step)")
    print("="*70)
    
    model = LogisticRegression(
        num_classes=10,
        width_factor=4,
        depth=64,
        in_channels=1,
        img_size=28
    )
    
    # Create optimizer with adaptive LR (as in the fix)
    base_lr = 0.02
    depth = 64
    lr_scale = 1.0 / np.sqrt(depth / 32.0)
    adaptive_lr = base_lr * lr_scale
    
    print(f"   Using adaptive LR: {base_lr:.4f} -> {adaptive_lr:.4f} (scale: {lr_scale:.3f})")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=adaptive_lr, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create dummy batch
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, 10, (batch_size,))
    
    # Training step
    try:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        
        print(f"   Loss: {loss.item():.4f}")
        
        loss.backward()
        
        # Check gradients
        has_nan_grad = False
        max_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"❌ FAILED: NaN/Inf in gradients of {name}")
                    has_nan_grad = True
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
        
        print(f"   Max gradient norm: {max_grad_norm:.3f}")
        
        # Clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"   Total gradient norm (before clip): {total_norm:.3f}")
        
        optimizer.step()
        
        # Check weights after update
        has_nan_weights = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"❌ FAILED: NaN/Inf in {name} after update")
                has_nan_weights = True
        
        if not has_nan_grad and not has_nan_weights:
            print("✅ PASSED: Backward pass and weight update successful")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Backward pass error: {e}")
        return False


def test_config_validation():
    """Test configuration validator"""
    print("\n" + "="*70)
    print("TEST 4: Configuration Validation")
    print("="*70)
    
    config = {
        'dataset': 'mnist',
        'model_type': 'lr',
        'width_factor': 4,
        'depth': 64,
        'poison_ratio': 0.3,
        'lr': 0.02,
        'max_grad_norm': 1.0
    }
    
    is_valid, warnings = validate_config(config, verbose=True)
    
    if warnings:
        print("\n💡 Suggested fixes:")
        suggestions = suggest_fixes(config)
        for key, value in suggestions.items():
            if not key.endswith('_reason'):
                print(f"   {key}: {config.get(key)} -> {value}")
        print("✅ PASSED: Validator detected issues and provided suggestions")
        return True
    else:
        print("❌ FAILED: Validator should have detected issues with this config")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING TRAINING STABILITY FIXES")
    print("Configuration: depth=64, width_factor=4, model_type='lr'")
    print("="*70)
    
    results = {
        "Initialization": test_model_initialization(),
        "Forward Pass": test_forward_pass(),
        "Backward Pass": test_backward_pass(),
        "Config Validation": test_config_validation()
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        all_passed = all_passed and passed
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Fixes are working correctly!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Please review the output above")
        sys.exit(1)
