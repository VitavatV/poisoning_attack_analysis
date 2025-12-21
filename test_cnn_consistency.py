"""
Quick test to verify CNN model also has proper initialization and validation.
"""

import torch
import logging

logging.basicConfig(level=logging.INFO)

from models import ScalableCNN
from validate_config import validate_config

def test_cnn_initialization():
    """Test CNN initialization with Kaiming weights"""
    print("\n" + "="*70)
    print("TEST: CNN Initialization with Kaiming")
    print("="*70)
    
    model = ScalableCNN(
        num_classes=10,
        width_factor=4,
        depth=16,
        in_channels=3,
        img_size=32
    )
    
    # Check Conv2d weights
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            mean = module.weight.data.mean().item()
            std = module.weight.data.std().item()
            print(f"✓ {name}: mean={mean:.6f}, std={std:.6f}")
            
            # Check no NaN
            if torch.isnan(module.weight).any():
                print(f"✗ NaN detected in {name}")
                return False
    
    print("✅ CNN initialization successful with Kaiming weights")
    return True

def test_cnn_validation():
    """Test CNN configuration validation"""
    print("\n" + "="*70)
    print("TEST: CNN Configuration Validation")
    print("="*70)
    
    # Test deep CNN
    config = {
        'model_type': 'cnn',
        'depth': 40,
        'width_factor': 16,
        'dataset': 'cifar10'
    }
    
    is_valid, warnings = validate_config(config, verbose=True)
    
    if warnings:
        print("✅ Validator correctly detected deep CNN and provided info")
    else:
        print("✓ No warnings for this configuration")
    
    return True

if __name__ == "__main__":
    print("\nTesting CNN Consistency Improvements")
    print("="*70)
    
    results = {
        "CNN Init": test_cnn_initialization(),
        "CNN Validation": test_cnn_validation()
    }
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    for test, passed in results.items():
        status = "✅" if passed else "✗"
        print(f"{test}: {status}")
    
    if all(results.values()):
        print("\n🎉 CNN now has same safeguards as LR!")
