import yaml
import itertools
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split

# Import modules ของเรา
from models import ScalableCNN
from data_utils import load_global_dataset, partition_data_dirichlet, get_client_dataloader
from Experiment101.utils import train_client, evaluate_model, fed_avg, EarlyStopping
from utils import train_client, evaluate_model, fed_avg, fed_median, EarlyStopping

import pandas as pd # เพิ่ม import
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_experiments(phase_config, defaults):
    # (โค้ดเดิมสำหรับแตก Grid Search)
    vary_params = {}
    for item in phase_config['combinations']:
        vary_params.update(item)
    keys = list(vary_params.keys())
    vals = list(vary_params.values())
    
    experiments = []
    for instance in itertools.product(*vals):
        exp_setup = defaults.copy()
        exp_setup['dataset'] = phase_config['dataset']
        exp_setup['phase_name'] = phase_config.get('phase_name', 'Unknown')
        for k, v in zip(keys, instance):
            exp_setup[k] = v
        experiments.append(exp_setup)
    return experiments

def run_single_experiment(config):
    print("\n" + "="*60)
    print(f"RUNNING: {config['phase_name']}")
    print(f"Setting: Dataset={config['dataset']}, Alpha={config['alpha']}")
    print(f"Model: W={config['width_factor']}, D={config['depth']}")
    print(f"Poison: {config['poison_ratio']} ({config['data_ordering']})")
    print("="*60)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Data
    train_ds_full, test_ds = load_global_dataset(config['dataset'])
    
    # --- Validation Split for Early Stopping ---
    val_size = int(len(train_ds_full) * config['validation_split'])
    train_size = len(train_ds_full) - val_size
    
    # ใช้ Seed เพื่อให้ Reproducible
    train_ds, val_ds = random_split(train_ds_full, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(42))
    
    # DataLoader สำหรับ Server ไว้ Valid/Test
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    
    # Partition ข้อมูล Train (ส่วนที่เหลือ) ให้ Clients
    client_indices = partition_data_dirichlet(train_ds_full, config['num_clients'], 
                                              alpha=config['alpha'], seed=42)
    # หมายเหตุ: client_indices ตอนนี้อิง index ของ train_ds_full ซึ่งถูกต้องแล้ว 
    # แต่ต้องระวังไม่ให้ index ของ val_ds หลุดไปหา Client 
    # (ในโค้ด partition_data_dirichlet ควรแก้ให้รับ subset หรือ handle index mapping ถ้าจะให้เป๊ะ 100% 
    # แต่เพื่อความง่าย เราจะ partition 'train_ds' ที่เป็น subset แทน)
    
    # **Correction:** Partition บน subset 'train_ds' เพื่อความถูกต้องของข้อมูล
    # ดึง indices ของ train_ds ออกมา
    train_subset_indices = train_ds.indices
    # สร้าง Sub-dataset ชั่วคราวเพื่อให้ partitioner ทำงานง่าย
    train_ds_only = Subset(train_ds_full, train_subset_indices)
    client_indices_subset = partition_data_dirichlet(train_ds_only, config['num_clients'], 
                                                     alpha=config['alpha'])
    
    # Map subset indices กลับไปเป็น Global Indices (สำคัญมาก)
    client_global_indices = {}
    for cid, idxs in client_indices_subset.items():
        client_global_indices[cid] = [train_subset_indices[i] for i in idxs]

    # 2. Setup Global Model
    num_classes = 100 if config['dataset'] == 'cifar100' else 10
    global_model = ScalableCNN(
        num_classes=num_classes, 
        width_factor=config['width_factor'], 
        depth=config['depth']
    ).to(device)
    
    global_weights = global_model.state_dict()
    
    # 3. Setup Early Stopping
    early_stopper = EarlyStopping(
        patience=config['early_stopping_patience'], 
        min_delta=config['min_delta']
    )
    
    # 4. Global Training Loop
    best_acc = 0.0
    
    for round_idx in range(config['global_rounds']):
        local_weights = []
        
        # Select Clients (ใช้ทั้งหมด หรือสุ่มตาม fraction_fit)
        m = max(int(config['fraction_fit'] * config['num_clients']), 1)
        selected_clients = np.random.choice(range(config['num_clients']), m, replace=False)
        
        # --- Local Training ---
        for client_id in selected_clients:
            # ตรวจสอบว่าเป็น Attacker หรือไม่
            # สมมติ Client 0-2 เป็น Attacker (30% ถ้า poison_ratio=0.3 และกระจายแบบสุ่ม)
            # ใน Paper นี้เรากำหนดให้ Poison Ratio เป็น "ปริมาณข้อมูลพิษใน dataset รวม" 
            # หรือ "เปอร์เซ็นต์ clients ที่เป็น attacker"
            # เพื่อให้ตรงกับ H2, H4.2: เราจะกำหนดให้ Client ทุกคนมีโอกาสโดน Poison 
            # ตาม config['poison_ratio'] ในระดับ Data Sample (Data Poisoning)
            
            # Setup Config เฉพาะ Client
            client_config = config.copy()
            # is_attacker=True เพื่อเปิดใช้งาน Poison logic ใน get_client_dataloader
            is_victim = True # ทุกคนมีสิทธิ์โดน Poison ตาม Ratio
            
            train_loader = get_client_dataloader(
                train_ds_full, 
                client_global_indices[client_id], 
                client_config, 
                is_attacker=is_victim 
            )
            
            # Init Model with Global Weights
            local_model = copy.deepcopy(global_model)
            local_model.load_state_dict(global_weights)
            
            # Train
            w = train_client(local_model, train_loader, 
                             epochs=config['local_epochs'], 
                             lr=config['lr'], 
                             device=device,
                             momentum=config.get('momentum', 0.9))
            local_weights.append(w)
            
        # --- Aggregation ---
        aggregator_name = config.get('aggregator', 'fedavg')
        if aggregator_name == 'median':
            global_weights = fed_median(local_weights)
        else:
            global_weights = fed_avg(local_weights)
        global_model.load_state_dict(global_weights)
        
        # --- Validation & Early Stopping ---
        val_loss, val_acc = evaluate_model(global_model, val_loader, device)
        
        print(f"Round {round_idx+1}/{config['global_rounds']} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Check Early Stopping
        early_stopper(val_loss, global_weights)
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        if early_stopper.early_stop:
            print(">>> Early Stopping Triggered!")
            # Load best weights back
            global_model.load_state_dict(early_stopper.get_best_weights())
            break
            
    # 5. Final Test
    test_loss, test_acc = evaluate_model(global_model, test_loader, device)
    print(f"FINAL RESULT: Test Acc: {test_acc:.4f}")
    
    # TODO: Save results to CSV/JSON here
    
def main():
    config = load_config()
    defaults = config['defaults']
    
    phases = [
        ('Phase 1: Landscape', config['phase1_landscape']),
        ('Phase 2: Mechanism', config['phase2_mechanism']),
        ('Phase 3: Validation', config['phase3_validation'])
    ]
    
    # 1. เตรียม List สำหรับเก็บผลลัพธ์ทั้งหมด
    all_results = []
    
    for phase_name, phase_cfg in phases:
        phase_cfg['phase_name'] = phase_name
        exp_list = generate_experiments(phase_cfg, defaults)
        
        for exp in exp_list:
            # รันการทดลอง (สมมติว่า run_single_experiment return ค่า acc กลับมาด้วย)
            # คุณต้องแก้ run_single_experiment ให้ return test_acc, test_loss, val_acc
            # ตัวอย่าง: test_loss, test_acc, best_val_acc = run_single_experiment(exp)
            
            # (Mockup: เพื่อให้ code รันผ่านในตัวอย่างนี้ ผมจะสมมติค่า return)
            # ใน code จริงให้แก้ run_single_experiment ให้ return ค่าจริงออกมา
            # test_loss, test_acc = evaluate_model(...) 
            # return test_loss, test_acc
            
            # --- จำลองการเก็บผล (คุณแก้ให้รับค่าจริง) ---
            test_loss, test_acc = 0.5, 0.85 # สมมติ
            best_val_acc = 0.86             # สมมติ
            # ----------------------------------------

            # 2. รวบรวมข้อมูลลง Dictionary
            result_entry = {
                "phase": phase_name,
                "dataset": exp['dataset'],
                "width_factor": exp['width_factor'],
                "depth": exp['depth'],
                "poison_ratio": exp['poison_ratio'],
                "alpha": exp['alpha'],
                "data_ordering": exp['data_ordering'],
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_val_acc": best_val_acc
            }
            all_results.append(result_entry)
            
            # 3. Save ระหว่างทาง (กันคอมค้างแล้วข้อมูลหาย)
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(defaults['output_dir'], "final_results.csv"), index=False)

    print("All experiments completed. Results saved to final_results.csv")

if __name__ == "__main__":
    main()