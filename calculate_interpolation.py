"""
คำนวณจุด Interpolation Threshold สำหรับ MNIST และ CIFAR10
ตามสูตร: P = n × k (จำนวนพารามิเตอร์ = จำนวนตัวอย่าง × จำนวนคลาส)
"""

import sys
sys.path.insert(0, 'c:/github/poisoning_attack_analysis')
from models import ScalableCNN

# ข้อมูลจากรูป
DATASETS = {
    'MNIST': {
        'n_train': 600000,  # จำนวนตัวอย่าง train
        'img_size': 28,
        'in_channels': 1,
        'k': 10,  # จำนวนคลาส
        'interpolation_threshold': 600000 * 10  # = 6,000,000
    },
    'CIFAR10': {
        'n_train': 50000,
        'img_size': 32,
        'in_channels': 3,
        'k': 10,
        'interpolation_threshold': 50000 * 10  # = 500,000
    }
}

def count_parameters(model):
    """นับจำนวนพารามิเตอร์ทั้งหมดในโมเดล"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_interpolation_thresholds():
    """วิเคราะห์หา width และ depth ที่ใกล้เคียงจุด interpolation threshold"""
    
    # ทดสอบค่า width และ depth ต่างๆ
    width_factors = [64, 96, 128, 192, 256, 384, 448, 480, 512]
    depths = [4]
    
    print("=" * 100)
    print("INTERPOLATION THRESHOLD ANALYSIS")
    print("=" * 100)
    
    for dataset_name, dataset_info in DATASETS.items():
        print(f"\n{'='*100}")
        print(f"Dataset: {dataset_name}")
        print(f"  Training samples (n): {dataset_info['n_train']:,}")
        print(f"  Classes (k): {dataset_info['k']}")
        print(f"  Interpolation threshold (P = n × k): {dataset_info['interpolation_threshold']:,} parameters")
        print(f"{'='*100}\n")
        
        results = []
        
        for depth in depths:
            for width in width_factors:
                try:
                    model = ScalableCNN(
                        num_classes=dataset_info['k'],
                        width_factor=width,
                        depth=depth,
                        in_channels=dataset_info['in_channels'],
                        img_size=dataset_info['img_size']
                    )
                    
                    n_params = count_parameters(model)
                    threshold = dataset_info['interpolation_threshold']
                    ratio = n_params / threshold
                    
                    # เก็บผลลัพธ์
                    results.append({
                        'width': width,
                        'depth': depth,
                        'n_params': n_params,
                        'ratio': ratio,
                        'diff': abs(n_params - threshold)
                    })
                    
                except Exception as e:
                    continue
        
        # เรียงตามความใกล้เคียงกับ threshold
        results.sort(key=lambda x: x['diff'])
        
        print(f"🎯 Top 10 configurations closest to interpolation threshold:\n")
        print(f"{'Width':<8} {'Depth':<8} {'Parameters':<15} {'Ratio (P/nk)':<15} {'Status':<20}")
        print("-" * 100)
        
        for i, r in enumerate(results[:10]):
            status = ""
            if r['ratio'] < 1.0:
                status = "Under-param"
            elif 0.9 <= r['ratio'] <= 1.1:
                status = "⭐ INTERPOLATION"
            else:
                status = "Over-param"
            
            print(f"{r['width']:<8} {r['depth']:<8} {r['n_params']:<15,} {r['ratio']:<15.4f} {status:<20}")
        
        # หาค่าที่ใกล้ที่สุด
        best = results[0]
        print(f"\n✨ BEST MATCH:")
        print(f"   Width Factor: {best['width']}")
        print(f"   Depth: {best['depth']}")
        print(f"   Parameters: {best['n_params']:,}")
        print(f"   Ratio: {best['ratio']:.4f} (1.0 = perfect interpolation)")
        
        # แสดงช่วง regime ต่างๆ
        print(f"\n📊 Regime Analysis:")
        under = [r for r in results if r['ratio'] < 0.9]
        inter = [r for r in results if 0.9 <= r['ratio'] <= 1.1]
        over = [r for r in results if r['ratio'] > 1.1]
        
        print(f"   Under-parameterized (P < 0.9nk): {len(under)} configs")
        print(f"   Interpolation regime (0.9nk ≤ P ≤ 1.1nk): {len(inter)} configs")
        print(f"   Over-parameterized (P > 1.1nk): {len(over)} configs")

if __name__ == "__main__":
    analyze_interpolation_thresholds()

# INTERPOLATION THRESHOLD ANALYSIS
# ====================================================================================================

# ====================================================================================================
# Dataset: MNIST
#   Training samples (n): 600,000
#   Classes (k): 10
#   Interpolation threshold (P = n × k): 6,000,000 parameters
# ====================================================================================================

# 🎯 Top 10 configurations closest to interpolation threshold:

# Width    Depth    Parameters      Ratio (P/nk)    Status
# ----------------------------------------------------------------------------------------------------
# 448      4        5,647,946       0.9413          Under-param
# 480      4        6,466,090       1.0777          ⭐ INTERPOLATION
# 512      4        7,339,530       1.2233          Over-param
# 384      4        4,177,546       0.6963          Under-param
# 256      4        1,900,298       0.3167          Under-param
# 192      4        1,093,450       0.1822          Under-param
# 128      4        507,786         0.0846          Under-param
# 96       4        297,898         0.0496          Under-param
# 64       4        143,306         0.0239          Under-param

# ✨ BEST MATCH:
#    Width Factor: 448
#    Depth: 4
#    Parameters: 5,647,946
#    Ratio: 0.9413 (1.0 = perfect interpolation)

# 📊 Regime Analysis:
#    Under-parameterized (P < 0.9nk): 6 configs
#    Interpolation regime (0.9nk ≤ P ≤ 1.1nk): 2 configs
#    Over-parameterized (P > 1.1nk): 1 configs

# ====================================================================================================
# Dataset: CIFAR10
#   Training samples (n): 50,000
#   Classes (k): 10
#   Interpolation threshold (P = n × k): 500,000 parameters
# ====================================================================================================

# 🎯 Top 10 configurations closest to interpolation threshold:

# Width    Depth    Parameters      Ratio (P/nk)    Status
# ----------------------------------------------------------------------------------------------------
# 128      4        529,290         1.0586          ⭐ INTERPOLATION
# 96       4        314,026         0.6281          Under-param
# 64       4        154,058         0.3081          Under-param
# 192      4        1,125,706       2.2514          Over-param
# 256      4        1,943,306       3.8866          Over-param
# 384      4        4,242,058       8.4841          Over-param
# 448      4        5,723,210       11.4464         Over-param
# 480      4        6,546,730       13.0935         Over-param
# 512      4        7,425,546       14.8511         Over-param

# ✨ BEST MATCH:
#    Width Factor: 128
#    Depth: 4
#    Parameters: 529,290
#    Ratio: 1.0586 (1.0 = perfect interpolation)

# 📊 Regime Analysis:
#    Under-parameterized (P < 0.9nk): 2 configs
#    Interpolation regime (0.9nk ≤ P ≤ 1.1nk): 1 configs
#    Over-parameterized (P > 1.1nk): 6 configs