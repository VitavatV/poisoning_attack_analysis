import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def partition_data_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    """
    แบ่งข้อมูลแบบ Non-IID ตามการแจกแจง Dirichlet
    
    Args:
        dataset: PyTorch Dataset (เช่น MNIST) ที่มี attribute .targets หรือ .classes
        num_clients (int): จำนวนไคลเอนต์ (N)
        alpha (float): ค่าความเข้มข้น (Concentration parameter). 
                       ค่าน้อย (0.1) = Non-IID สูงมาก
                       ค่ามาก (100) = ใกล้เคียง IID
        seed (int): เพื่อให้ผลการแบ่งเหมือนเดิมทุกครั้ง (Reproducibility)
        
    Returns:
        dict: dictionary ที่ key คือ client_id และ value คือ list ของ index ข้อมูล
    """
    np.random.seed(seed)
    
    # ดึง Label ทั้งหมดออกมาจาก Dataset
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'y'): # บาง Dataset อาจใช้ชื่อนี้
        labels = np.array(dataset.y)
    else:
        # กรณี List ธรรมดา
        labels = np.array([y for _, y in dataset])
        
    num_classes = len(np.unique(labels))
    min_size = 0
    client_idcs = {i: [] for i in range(num_clients)}
    
    # วนลูปจนกว่าไคลเอนต์ทุกรายจะมีข้อมูล (ป้องกันกรณีสุ่มแล้วได้ 0)
    while min_size < 10:
        client_idcs = {i: [] for i in range(num_clients)}
        
        # วนลูปทีละคลาส (Class-wise selection)
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            # สุ่มสัดส่วนการกระจายของคลาส k ให้กับ N clients
            # ตามสมการ p_k ~ Dir(alpha)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # ปรับสัดส่วนให้สมดุลกับจำนวนข้อมูลในคลาสนั้นๆ เพื่อไม่ให้ index เกิน
            proportions = np.array([p * (len(idx_k) < num_clients and 1/num_clients or 1) for p in proportions])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # แบ่ง index ตามสัดส่วนที่สุ่มได้
            client_splits = np.split(idx_k, proportions)
            
            # ใส่ข้อมูลลงใน dict ของแต่ละ client
            for i in range(num_clients):
                client_idcs[i] += client_splits[i].tolist()
                
        # ตรวจสอบว่าไคลเอนต์ที่ได้ข้อมูลน้อยที่สุด มีข้อมูลกี่ตัว
        min_size = min([len(client_idcs[i]) for i in range(num_clients)])
        
    return client_idcs

# --- ส่วนสำหรับการแสดงผล (Visualization) ---
def plot_distribution(client_idcs, labels, num_classes, num_clients, alpha):
    """สร้างกราฟแท่งแสดงการกระจายของคลาสในแต่ละไคลเอนต์"""
    plt.figure(figsize=(10, 6))
    
    # เตรียมข้อมูลสำหรับ plot
    client_class_counts = np.zeros((num_clients, num_classes))
    for client_id, indices in client_idcs.items():
        client_labels = labels[indices]
        counts = np.bincount(client_labels, minlength=num_classes)
        client_class_counts[client_id] = counts
        
    # Plot Stacked Bar Chart
    bottom = np.zeros(num_clients)
    for k in range(num_classes):
        plt.bar(range(num_clients), client_class_counts[:, k], bottom=bottom, label=f'Class {k}')
        bottom += client_class_counts[:, k]
        
    plt.title(f'Data Distribution per Client (Alpha={alpha})')
    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Classes')
    plt.tight_layout()
    plt.show()