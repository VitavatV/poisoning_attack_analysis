import torch
import yaml
from data_utils import load_global_dataset, partition_data_dirichlet, get_client_dataloader
from torch.utils.data import TensorDataset

def log(msg):
    with open("test_results.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

def test_memory_loading():
    # Clear previous results
    with open("test_results.txt", "w") as f:
        f.write("Starting test...\n")
        
    log("Testing memory loading...")
    
    # Mock config
    config = {
        'batch_size': 10,
        'load_to_memory': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'poison_ratio': 0.0,
        'data_ordering': 'shuffle',
        'num_clients': 2,
        'alpha': 100.0
    }
    
    log(f"Device: {config['device']}")
    
    # Load small dataset (MNIST)
    train_ds, _ = load_global_dataset("mnist")
    
    # Partition
    client_indices = partition_data_dirichlet(train_ds, config['num_clients'], config['alpha'])
    client_idx = client_indices[0][:100] # Take small subset
    
    # Get loader
    loader = get_client_dataloader(train_ds, client_idx, config)
    
    # Check if dataset is TensorDataset (which implies it was loaded to memory)
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
    test_memory_loading()
