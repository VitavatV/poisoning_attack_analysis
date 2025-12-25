import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

# --- 1. Custom Dataset Wrapper ---
class FederatedDataset(Dataset):
    """
    Wrapper class สำหรับจัดการข้อมูลของแต่ละ Client
    รองรับการเปลี่ยน Label (Poisoning) โดยไม่กระทบ Dataset หลัก
    """
    def __init__(self, global_dataset, indices, poison_map=None, transform=None):
        self.global_dataset = global_dataset
        self.indices = indices  # รายการ Index ที่ Client นี้ถือครอง
        self.poison_map = poison_map if poison_map else {} # Dict {index: new_label}
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # แปลงจาก Local Index -> Global Index
        global_idx = self.indices[idx]
        
        # ดึงข้อมูลจาก Global Dataset
        image, label = self.global_dataset[global_idx]
        
        # ตรวจสอบว่าข้อมูลนี้ถูก Poison หรือไม่
        is_poisoned = False
        if global_idx in self.poison_map:
            label = self.poison_map[global_idx] # Flip Label
            is_poisoned = True
            
        # Apply Transform (ถ้ามี)
        if self.transform:
            image = self.transform(image)
        elif hasattr(self.global_dataset, 'transform') and self.global_dataset.transform:
            # กรณี Dataset ดั้งเดิมมี transform อยู่แล้ว (เช่น MNIST)
            # ต้องระวังการ apply ซ้ำ หรือถ้าดึงมาเป็น Tensor แล้วก็ข้ามไป
            pass

        # Return flag is_poisoned เพื่อใช้ในการวิเคราะห์ผลได้ด้วย
        return image, label, is_poisoned

# --- 2. Dirichlet Partitioning (Non-IID) ---
def partition_data_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    """
    แบ่งข้อมูลให้ Clients ตามการแจกแจง Dirichlet
    """
    np.random.seed(seed)
    
    # ดึง Targets ออกมา
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'y'):
        labels = np.array(dataset.y)
    else:
        # Fallback สำหรับ Dataset ทั่วไป
        labels = np.array([y for _, y in dataset])
        
    num_classes = len(np.unique(labels))
    min_size = 0
    client_idcs = {}

    # วนลูปจนกว่าจะได้ partition ที่ไม่มี client ไหนว่างเปล่า
    while min_size < 10:
        client_idcs = {i: [] for i in range(num_clients)}
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance scaling เพื่อไม่ให้ index เกิน
            proportions = np.array([p * (len(idx_k) < num_clients and 1/num_clients or 1) for p in proportions])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            client_splits = np.split(idx_k, proportions)
            for i in range(num_clients):
                client_idcs[i] += client_splits[i].tolist()
        
        min_size = min([len(client_idcs[i]) for i in range(num_clients)])
    
    return client_idcs

# --- 3. Client-Level Poisoning Selection ---
def select_malicious_clients(num_clients: int, poison_ratio: float, seed: int):
    """
    Select which clients are malicious based on poison_ratio.
    
    Args:
        num_clients: Total number of clients
        poison_ratio: Percentage of clients to be malicious (0.0 to 1.0)
        seed: Random seed
    
    Returns:
        List of malicious client IDs
    """
    if poison_ratio <= 0.0:
        return []
    
    np.random.seed(seed)
    num_malicious = max(1, int(num_clients * poison_ratio))
    malicious_clients = np.random.choice(num_clients, num_malicious, replace=False).tolist()
    
    return malicious_clients


def apply_label_flip_poisoning(gt_label: int, num_classes: int = 10) -> int:
    """
    Apply complementary label flip.
    0→9, 1→8, 2→7, 3→6, 4→5, 5→4, 6→3, 7→2, 8→1, 9→0
    
    Args:
        gt_label: Ground truth label
        num_classes: Total number of classes
    
    Returns:
        Flipped label
    """
    return (num_classes - 1) - gt_label


def apply_random_noise_poisoning(gt_label: int, num_classes: int = 10, seed: int = None) -> int:
    """
    Apply random noise: select random class except the correct class.
    
    Args:
        gt_label: Ground truth label
        num_classes: Total number of classes
        seed: Random seed for reproducibility
    
    Returns:
        Random label (not equal to gt_label)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get all possible classes except the ground truth
    possible_labels = [i for i in range(num_classes) if i != gt_label]
    return np.random.choice(possible_labels)


# --- 4. Legacy Poisoning Logic (Deprecated) ---
def apply_poisoning(dataset, client_indices, poison_ratio, target_class, poison_label, seed=42):
    """Fixed version with better error handling"""
    random.seed(seed)
    np.random.seed(seed)
    
    poison_map = {}
    all_targets = np.array(dataset.targets)
    
    # **FIXED**: Better list comprehension
    candidates = [idx for idx in client_indices if all_targets[idx] == target_class]
    others = [idx for idx in client_indices if all_targets[idx] != target_class]

    if not candidates:
        print(f"Warning: No samples of target_class {target_class} found")
        return {}, list(client_indices), []
    
    # **FIXED**: Prevent num_poison > len(candidates)
    num_poison = min(int(len(candidates) * poison_ratio), len(candidates))
    selected_poison = np.random.choice(candidates, num_poison, replace=False)
    
    for idx in selected_poison:
        poison_map[idx] = poison_label
    
    poisoned_indices_list = list(selected_poison)
    clean_indices_list = [i for i in client_indices if i not in selected_poison]
    
    return poison_map, clean_indices_list, poisoned_indices_list

# --- 5. Main Loader Builder (The Mechanism) ---
def get_client_dataloader(dataset, client_indices, config, is_malicious_client=False):
    """
    สร้าง DataLoader ที่รองรับทั้ง Label Flip และ Random Noise
    Client-level poisoning: malicious clients poison 100% of their data
    """
    poison_map = {}
    clean_idxs = client_indices
    poison_idxs = []
    
    # Get dataset info for poisoning functions
    if hasattr(dataset, 'targets'):
        all_targets = np.array(dataset.targets)
    elif hasattr(dataset, 'y'):
        all_targets = np.array(dataset.y)
    else:
        all_targets = np.array([y for _, y in dataset])
    
    num_classes = len(np.unique(all_targets))
    
    # Check if this client is malicious
    if is_malicious_client:
        # Malicious clients poison 100% of their data
        poison_type = config.get('poison_type', 'label_flip')
        
        # Create poison map for ALL indices of this client
        for idx in client_indices:
            gt_label = all_targets[idx]
            
            if poison_type == 'label_flip':
                # Apply complementary label flip
                poison_map[idx] = apply_label_flip_poisoning(gt_label, num_classes)
            elif poison_type == 'random_noise':
                # Apply random noise
                seed = config.get('seed', 42) + idx  # Use index for deterministic randomness
                poison_map[idx] = apply_random_noise_poisoning(gt_label, num_classes, seed)
            else:
                # Default to label flip
                poison_map[idx] = apply_label_flip_poisoning(gt_label, num_classes)
        
        # All indices are poisoned for malicious clients
        poison_idxs = list(client_indices)
        clean_idxs = []
    
    # ส่วนจัดการ Ordering (เหมือนเดิม)
    final_indices = []
    shuffle_batch = True
    ordering = config.get('data_ordering', 'shuffle')
    
    if ordering == 'shuffle':
        final_indices = clean_idxs + poison_idxs
        shuffle_batch = True 
    elif ordering == 'good_bad':
        random.shuffle(clean_idxs)
        random.shuffle(poison_idxs)
        final_indices = clean_idxs + poison_idxs
        shuffle_batch = False 
    elif ordering == 'bad_good':
        random.shuffle(poison_idxs)
        random.shuffle(clean_idxs)
        final_indices = poison_idxs + clean_idxs
        shuffle_batch = False
        
    local_dataset = FederatedDataset(dataset, final_indices, poison_map=poison_map)
    
    # [NEW] Load to Memory Logic
    device_config = config.get('device', 'cpu')
    
    if config.get('load_to_memory', False):
        device = torch.device(device_config)
        # Pre-load all data to a single Tensor
        data_list = []
        target_list = []
        
        # Iterate once to load everything
        for i in range(len(local_dataset)):
            img, target, is_poisoned = local_dataset[i]
            data_list.append(img)
            target_list.append(torch.tensor(target))
            
        # Stack and move to device
        if len(data_list) > 0:
            all_data = torch.stack(data_list).to(device)
            all_targets = torch.stack(target_list).to(device)
            
            from torch.utils.data import TensorDataset
            local_dataset = TensorDataset(all_data, all_targets)
        
        num_workers = 0
        pin_memory = False
    else:
        # Use 4 workers for parallel data loading (reduces I/O bottleneck)
        num_workers = 4
        pin_memory = True if device_config != 'cpu' else False
    
    loader = DataLoader(
        local_dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True # Prevent batch size 1 error with BatchNorm
    )
    
    return loader

# --- Helper: Load Global Dataset ---
def load_global_dataset(name="cifar10", root="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize เบื้องต้น
    ])
    
    if name == "cifar10":
        train_data = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    elif name == "cifar100":
        train_data = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    elif name == "mnist":
        # MNIST ต้องระวังเรื่อง Channel (1 channel) ถ้า Model รับ 3 ต้องแปลง
        transform_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_data = datasets.MNIST(root=root, train=True, download=True, transform=transform_mnist)
        test_data = datasets.MNIST(root=root, train=False, download=True, transform=transform_mnist)
        
    return train_data, test_data

def apply_random_poisoning(dataset, client_indices, poison_ratio, seed=42):
    """
    Random Noise Attack: เปลี่ยน Label เป็น class อื่นแบบสุ่ม (Untargeted)
    """
    np.random.seed(seed)  # ADD THIS
    poison_map = {}
    
    all_targets = np.array(dataset.targets)
    num_classes = len(np.unique(all_targets))
    
    # สุ่มเลือก Index ที่จะโดน Poison
    num_poison = int(len(client_indices) * poison_ratio)
    selected_poison = np.random.choice(client_indices, num_poison, replace=False)
    
    for idx in selected_poison:
        true_label = all_targets[idx]
        # สุ่ม Label ใหม่ที่ไม่ใช่ตัวเดิม
        possible_labels = list(range(num_classes))
        possible_labels.remove(true_label)
        poison_map[idx] = np.random.choice(possible_labels)
        
    # คืนค่า format เดียวกับ targeted poisoning
    # (แยก list poison/clean เพื่อรองรับ ordering logic เดิม)
    poisoned_indices_list = list(selected_poison)
    clean_indices_list = [i for i in client_indices if i not in selected_poison]
        
    return poison_map, clean_indices_list, poisoned_indices_list