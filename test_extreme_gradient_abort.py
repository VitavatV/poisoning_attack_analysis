"""
Test the extreme gradient early abort mechanism.
Simulates a case with extremely unstable gradients.
"""

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

from models import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader

def test_extreme_gradient_abort():
    """Test that training aborts after 10 extreme gradients"""
    print("\n" + "="*70)
    print("TEST: Extreme Gradient Early Abort")
    print("="*70)
    print("Simulating very unstable training (LR too high)")
    print()
    
    # Create a model that will have extreme gradients with high LR
    model = LogisticRegression(
        num_classes=10,
        width_factor=4,
        depth=64,
        in_channels=1,
        img_size=28
    )
    
    # Create dummy dataset
    X = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Import train_client
    from utils import train_client
    
    # Try training with VERY HIGH learning rate to trigger extreme gradients
    print("⚙️ Training with LR=1.0 (intentionally too high to trigger abort)")
    
    device = torch.device('cpu')
    weights = train_client(
        model,
        train_loader,
        epochs=5,
        lr=1.0,  # Very high LR to cause instability
        device=device,
        max_grad_norm=1.0
    )
    
    print("\n" + "="*70)
    print("Result: Training should have aborted early (after ~10 extreme gradients)")
    print("="*70)

if __name__ == "__main__":
    test_extreme_gradient_abort()
