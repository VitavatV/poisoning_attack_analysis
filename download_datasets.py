"""
Download and verify datasets for experiments
"""

import os
import torch
from torchvision import datasets, transforms

def download_datasets():
    """Download MNIST and CIFAR-10 datasets"""
    
    root = "./data"
    os.makedirs(root, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("="*60)
    print("Downloading Datasets")
    print("="*60)
    
    # Download MNIST
    print("\n[1/2] Downloading MNIST...")
    try:
        mnist_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        print(f"✓ MNIST downloaded successfully")
        print(f"  Train: {len(mnist_train)} samples")
        print(f"  Test:  {len(mnist_test)} samples")
    except Exception as e:
        print(f"✗ MNIST download failed: {e}")
        return False
    
    # Download CIFAR-10
    print("\n[2/2] Downloading CIFAR-10...")
    try:
        cifar_train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        cifar_test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        print(f"✓ CIFAR-10 downloaded successfully")
        print(f"  Train: {len(cifar_train)} samples")
        print(f"  Test:  {len(cifar_test)} samples")
    except Exception as e:
        print(f"✗ CIFAR-10 download failed: {e}")
        print(f"\nTrying alternative download method...")
        
        # Alternative: Use direct download with retry
        import urllib.request
        import tarfile
        
        cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        cifar_dir = os.path.join(root, "cifar-10-batches-py")
        
        if not os.path.exists(cifar_dir):
            try:
                print(f"Downloading from: {cifar_url}")
                tar_path = os.path.join(root, "cifar-10-python.tar.gz")
                
                urllib.request.urlretrieve(cifar_url, tar_path)
                print("✓ Download complete, extracting...")
                
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=root)
                
                os.remove(tar_path)
                print("✓ CIFAR-10 extracted successfully")
                
                # Try loading again
                cifar_train = datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
                cifar_test = datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
                print(f"✓ CIFAR-10 loaded successfully")
                print(f"  Train: {len(cifar_train)} samples")
                print(f"  Test:  {len(cifar_test)} samples")
            except Exception as e2:
                print(f"✗ Alternative download also failed: {e2}")
                return False
        else:
            print(f"✓ CIFAR-10 directory already exists")
    
    print("\n" + "="*60)
    print("✓ All datasets ready!")
    print("="*60)
    
    # Test loading
    print("\nTesting dataset loading...")
    try:
        sample = mnist_train[0]
        print(f"✓ MNIST sample shape: {sample[0].shape}")
        
        sample = cifar_train[0]
        print(f"✓ CIFAR-10 sample shape: {sample[0].shape}")
    except Exception as e:
        print(f"✗ Error loading samples: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_datasets()
    
    if success:
        print("\n✓ Ready to run experiments!")
        exit(0)
    else:
        print("\n✗ Dataset download failed. Please check your internet connection.")
        exit(1)
