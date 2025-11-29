import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ตั้งค่า Style กราฟให้ดูเป็นวิชาการ (Academic Style)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif' # ใช้ Font แบบพวก Times New Roman

RESULT_FILE = "./results/final_results.csv"
OUTPUT_DIR = "./results/plots"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    if not os.path.exists(RESULT_FILE):
        print(f"Error: File {RESULT_FILE} not found. Please run experiments first.")
        return None
    return pd.read_csv(RESULT_FILE)

def plot_phase1_landscape(df):
    """
    สร้างกราฟ Double Descent:
    X-axis: Width Factor (Model Complexity)
    Y-axis: Test Accuracy
    Line 1: Clean (Poison 0.0)
    Line 2: Poisoned (Poison 0.3)
    """
    print("Generating Plot: Phase 1 Landscape (Double Descent)...")
    
    # Filter เฉพาะ Phase 1
    data = df[df['phase'].str.contains("Phase 1")]
    if data.empty: return

    plt.figure(figsize=(8, 6))
    
    # Plot โดยแยกสีตาม Poison Ratio
    sns.lineplot(
        data=data, 
        x="width_factor", 
        y="test_acc", 
        hue="poison_ratio", 
        style="alpha",       # เส้นประ/เส้นทึบ แยก IID/Non-IID
        markers=True, 
        dashes=True,
        palette="viridis",
        linewidth=2.5,
        markersize=8
    )

    plt.title("Impact of Model Width on Robustness (Double Descent)", fontsize=14, weight='bold')
    plt.xlabel("Model Width Factor (Complexity)", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.legend(title="Poison Ratio / Alpha", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate Critical Regime (สมมติ Width=10 คือ Critical)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.3)
    plt.text(10.5, data['test_acc'].min(), 'Critical Regime', color='red', rotation=90)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase1_double_descent.png", dpi=300)
    plt.close()

def plot_phase2_mechanism(df):
    """
    สร้างกราฟแท่งเปรียบเทียบ Data Ordering (Mechanism)
    เปรียบเทียบ: Shuffle vs Good>Bad vs Bad>Good
    """
    print("Generating Plot: Phase 2 Mechanism...")
    
    data = df[df['phase'].str.contains("Phase 2")]
    if data.empty: return

    plt.figure(figsize=(7, 6))
    
    # Bar Chart
    ax = sns.barplot(
        data=data,
        x="data_ordering",
        y="test_acc",
        palette="magma",
        edgecolor="black"
    )

    plt.title("Effect of Data Ordering on Model Robustness", fontsize=14, weight='bold')
    plt.xlabel("Ordering Strategy", fontsize=12)
    plt.ylabel("Test Accuracy (under Attack)", fontsize=12)
    plt.ylim(0, 1.0) # Accuracy 0-1

    # ใส่ตัวเลขบนแท่งกราฟ
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase2_ordering_mechanism.png", dpi=300)
    plt.close()

def generate_summary_report(df):
    """
    สร้าง Text File สรุปผลการทดลอง
    """
    print("Generating Summary Report...")
    report_path = f"{OUTPUT_DIR}/experiment_summary.txt"
    
    with open(report_path, "w") as f:
        f.write("=== FEDERATED LEARNING EXPERIMENT SUMMARY ===\n\n")
        
        # 1. Phase 1 Summary
        f.write("--- PHASE 1: LANDSCAPE (Width vs Poison) ---\n")
        p1 = df[df['phase'].str.contains("Phase 1")]
        if not p1.empty:
            summary = p1.groupby(['width_factor', 'poison_ratio', 'alpha'])['test_acc'].mean()
            f.write(summary.to_string())
            f.write("\n\n")
            
        # 2. Phase 2 Summary
        f.write("--- PHASE 2: MECHANISM (Ordering) ---\n")
        p2 = df[df['phase'].str.contains("Phase 2")]
        if not p2.empty:
            summary = p2.groupby(['data_ordering'])['test_acc'].mean()
            f.write(summary.to_string())
            f.write("\n\n")
            
        # 3. Phase 3 Summary
        f.write("--- PHASE 3: VALIDATION (CIFAR-100) ---\n")
        p3 = df[df['phase'].str.contains("Phase 3")]
        if not p3.empty:
            summary = p3.groupby(['dataset', 'width_factor'])['test_acc'].mean()
            f.write(summary.to_string())
            f.write("\n\n")

    print(f"Summary saved to {report_path}")

def main():
    ensure_dir(OUTPUT_DIR)
    
    # 1. Load Data
    df = load_data()
    if df is None: return

    # 2. Generate Plots
    plot_phase1_landscape(df)
    plot_phase2_mechanism(df)
    
    # 3. Generate Text Report
    generate_summary_report(df)
    
    print("\nVisualization Complete! Check ./results/plots folder.")

if __name__ == "__main__":
    main()