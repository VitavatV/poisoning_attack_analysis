import os
from torchvision import datasets

def save_images(dataset, root_dir):
    for idx, (img, label) in enumerate(dataset):
        class_dir = os.path.join(root_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{idx}.png")
        img.save(img_path)

def main():
    base_dir = "./data/mnist_images"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # No transform needed; MNIST returns PIL images by default
    train_dataset = datasets.MNIST(root="./data", train=True, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True)

    print("Saving training images...")
    save_images(train_dataset, train_dir)
    print("Saving testing images...")
    save_images(test_dataset, test_dir)
    print("Done.")

if __name__ == "__main__":
    main()