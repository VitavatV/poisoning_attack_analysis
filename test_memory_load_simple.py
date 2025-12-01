import torch
from torch.utils.data import Dataset, TensorDataset
from data_utils import get_client_dataloader

class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 1, 28, 28)
        self.targets = torch.randint(0, 10, (size,))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def log(msg):
    with open("test_results_simple.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

def test_memory_loading_simple():
    with open("test_results_simple.txt", "w") as f:
        f.write("Starting simple test...\n")
        
    log("Testing memory loading with dummy data...")
    
    # Mock config
    config = {
        'batch_size': 10,
        'load_to_memory': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'poison_ratio': 0.0,
        'data_ordering': 'shuffle',
        'num_clients': 1,
        'alpha': 100.0
    }
    
    log(f"Device: {config['device']}")
    
    # Create dummy dataset
    train_ds = DummyDataset(size=100)
    
    # Client indices (all data)
    client_idx = list(range(100))
    
    # Get loader
    loader = get_client_dataloader(train_ds, client_idx, config)
    
    # Check if dataset is TensorDataset
    if isinstance(loader.dataset, TensorDataset):
        log("PASS: Dataset is TensorDataset")
    else:
        log(f"FAIL: Dataset is {type(loader.dataset)}")
        
    # Check device of data
    batch = next(iter(loader))
    images, labels = batch
    
    log(f"Images device: {images.device}")
    log(f"Labels device: {labels.device}")
    
    if str(images.device) == config['device'] or (config['device'] == 'cuda' and 'cuda' in str(images.device)):
        log("PASS: Data is on correct device")
    else:
        log(f"FAIL: Data is on {images.device}, expected {config['device']}")

if __name__ == "__main__":
    test_memory_loading_simple()
