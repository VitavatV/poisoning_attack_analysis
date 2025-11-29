import os
import torch
from torchvision import datasets, transforms

# กำหนด Path หลักที่จะเก็บข้อมูล
DATA_ROOT = "./data"

def download_and_verify(dataset_name):
    print(f"\n{'='*40}")
    print(f"Checking & Downloading: {dataset_name.upper()}")
    print(f"{'='*40}")

    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(DATA_ROOT, exist_ok=True)
    
    train_ds = None
    test_ds = None

    try:
        if dataset_name == 'cifar10':
            # CIFAR-10
            train_ds = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
            test_ds = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)
            
        elif dataset_name == 'cifar100':
            # CIFAR-100
            train_ds = datasets.CIFAR100(root=DATA_ROOT, train=True, download=True)
            test_ds = datasets.CIFAR100(root=DATA_ROOT, train=False, download=True)
            
        elif dataset_name == 'mnist':
            # MNIST
            train_ds = datasets.MNIST(root=DATA_ROOT, train=True, download=True)
            test_ds = datasets.MNIST(root=DATA_ROOT, train=False, download=True)
            
        elif dataset_name == 'tiny-imagenet':
            print("Note: Tiny-ImageNet is not directly supported by torchvision.datasets with auto-download.")
            print("Please download 'tiny-imagenet-200.zip' from http://cs231n.stanford.edu/")
            print("and extract it to ./data/tiny-imagenet-200 manually.")
            return

        # --- Verification Stats ---
        if train_ds and test_ds:
            print(f"\n[SUCCESS] {dataset_name.upper()} is ready.")
            print(f" - Train samples: {len(train_ds)}")
            print(f" - Test samples:  {len(test_ds)}")
            
            # ตรวจสอบ Shape ของภาพแรก
            img, label = train_ds[0]
            # แปลงเป็น Tensor เพื่อดู Shape (ถ้ายังไม่ใช่ Tensor)
            if not isinstance(img, torch.Tensor):
                transform = transforms.ToTensor()
                img = transform(img)
            
            print(f" - Image Shape:   {img.shape}")
            print(f" - Label Example: {label}")
            
            # ตรวจสอบจำนวนคลาส
            classes = train_ds.classes if hasattr(train_ds, 'classes') else train_ds.class_to_idx
            print(f" - Classes Count: {len(classes)}")

    except Exception as e:
        print(f"\n[ERROR] Failed to download {dataset_name}: {e}")

def main():
    # รายชื่อ Dataset ที่ต้องการใช้ในการทดลอง
    # ตามแผน Smart & Lean Design เราเน้น CIFAR-10 และ CIFAR-100
    target_datasets = ['cifar10', 'cifar100', 'mnist']

    for name in target_datasets:
        download_and_verify(name)

    print(f"\n{'='*40}")
    print("All downloads completed. You can now run 'experiment_runner.py'")
    print(f"Data is stored in: {os.path.abspath(DATA_ROOT)}")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()