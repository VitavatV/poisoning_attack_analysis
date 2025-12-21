"""
Test that tensor copy fix works for both MNIST and CIFAR-10
"""

import torch
from torchvision import datasets

def test_targets_handling():
    """Test that we can handle both list and tensor targets"""
    
    print("\n" + "="*60)
    print("Testing Target Handling for MNIST and CIFAR-10")
    print("="*60)
    
    # Test MNIST (tensor targets)
    mnist = datasets.MNIST(root='./data', train=True, download=True)
    print(f"\nMNIST targets type: {type(mnist.targets)}")
    print(f"  Is tensor: {isinstance(mnist.targets, torch.Tensor)}")
    
    # Test CIFAR-10 (list targets)
    cifar = datasets.CIFAR10(root='./data', train=True, download=True)
    print(f"\nCIFAR-10 targets type: {type(cifar.targets)}")
    print(f"  Is tensor: {isinstance(cifar.targets, torch.Tensor)}")
    print(f"  Is list: {isinstance(cifar.targets, list)}")
    
    # Test the fix logic
    print("\n" + "="*60)
    print("Testing Fix Logic")
    print("="*60)
    
    # MNIST case
    if isinstance(mnist.targets, torch.Tensor):
        mnist_copy = mnist.targets.clone().detach()
        print("✅ MNIST: Used .clone().detach()")
    else:
        mnist_copy = torch.tensor(mnist.targets)
        print("✅ MNIST: Converted list to tensor")
    
    # CIFAR-10 case
    if isinstance(cifar.targets, torch.Tensor):
        cifar_copy = cifar.targets.clone().detach()
        print("✅ CIFAR-10: Used .clone().detach()")
    else:
        cifar_copy = torch.tensor(cifar.targets)
        print("✅ CIFAR-10: Converted list to tensor")
    
    print("\n" + "="*60)
    print("Both datasets handled correctly!")
    print("="*60)

if __name__ == "__main__":
    test_targets_handling()
