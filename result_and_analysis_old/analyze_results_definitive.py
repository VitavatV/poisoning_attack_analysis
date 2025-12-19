import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast # ใช้แปลง string list "[...]" กลับเป็น list จริง

# ตั้งค่า Style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'

RESULT_FILE = "./results_definitive/final_results.csv"
OUTPUT_DIR = "./results_definitive/plots"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    if not os.path.exists(RESULT_FILE):
        print(f"Error: File {RESULT_FILE} not found.")
        return None
    return pd.read_csv(RESULT_FILE)

def plot_exp1_double_descent(df):
    """
    Exp 1: Fine-Grained Landscape
    Plot Mean Accuracy พร้อม Error Band (fill_between)
    """
    print("Generating Plot: Exp 1 Double Descent...")
    
    # Filter ข้อมูล Exp 1
    # หมายเหตุ: ต้องเช็คชื่อ phase ให้ตรงกับใน config_definitive.yaml
    data = df[df['phase'].str.contains("exp1", case=False, na=False)].copy()
    if data.empty: 
        print(" - No data for Exp 1")
        return

    plt.figure(figsize=(10, 6))
    
    # แยกเส้นตาม Poison Ratio
    poison_ratios = data['poison_ratio'].unique()
    colors = sns.color_palette("viridis", len(poison_ratios))

    for i, pr in enumerate(sorted(poison_ratios)):
        subset = data[data['poison_ratio'] == pr].sort_values("width_factor")
        
        # วาดเส้นกราฟ (Mean)
        plt.plot(subset['width_factor'], subset['mean_test_acc'], 
                 label=f"Poison Ratio {pr}", color=colors[i], marker='o', linewidth=2)
        
        # วาด Error Band (Mean +/- Std)
        plt.fill_between(
            subset['width_factor'], 
            subset['mean_test_acc'] - subset['std_test_acc'],
            subset['mean_test_acc'] + subset['std_test_acc'],
            color=colors[i], alpha=0.2
        )

    plt.title("Double Descent Landscape with Error Bars", fontsize=16, weight='bold')
    plt.xlabel("Model Width Factor", fontsize=14)
    plt.ylabel("Test Accuracy (Mean ± Std)", fontsize=14)
    plt.legend(title="Attack Intensity")
    
    # Highlight Critical Regime
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.4)
    plt.text(10.5, subset['mean_test_acc'].min(), 'Critical Regime', color='red', rotation=90)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp1_double_descent_definitive.png", dpi=300)
    plt.close()

def plot_exp2_defense(df):
    """
    Exp 2: Defense Comparison (FedAvg vs Median)
    Bar Chart พร้อม Error Bars
    """
    print("Generating Plot: Exp 2 Defense Comparison...")
    
    data = df[df['phase'].str.contains("exp2", case=False, na=False)].copy()
    if data.empty: 
        print(" - No data for Exp 2")
        return
    
    if 'aggregator' not in data.columns:
        print("Warning: 'aggregator' column not found in results")
        return

    plt.figure(figsize=(8, 6))
    
    # สร้าง Bar Chart
    # Hue: Aggregator (FedAvg vs Median)
    # X: Width Factor (Small vs Large)
    ax = sns.barplot(
        data=data,
        x="width_factor",
        y="mean_test_acc",
        hue="aggregator",
        palette="Set2",
        edgecolor="black",
        capsize=0.1, # เพิ่มขีด Error Bar ที่หัวแท่ง
        errorbar=None # เราจะวาด Error Bar เองจากค่า std
    )

    # วาด Error Bars ด้วยตัวเอง (เพราะ Seaborn รับค่า raw ไม่ได้รับ mean/std)
    # ต้องคำนวณตำแหน่ง x ของแต่ละแท่ง
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    
    # เรียงลำดับข้อมูลให้ตรงกับแท่งกราฟ (อาจซับซ้อนเล็กน้อยใน seaborn)
    # วิธีง่ายกว่า: ใช้ Loop วาด plt.errorbar ทับลงไป หรือใช้ yerr ใน pandas.plot
    # แต่เพื่อความง่ายและสวยงาม เราใช้เทคนิคนี้:
    
    # ... (ข้าม Technical Detail การวาด error bar บน grouped bar ที่ซับซ้อน) ...
    # ... (ใช้วิธีพล็อตแบบ Manual ง่ายกว่าสำหรับข้อมูล Summary) ...
    
    plt.close() # Reset
    
    # Plot ใหม่ด้วย Pandas Plotting (ง่ายกว่าสำหรับ Summary Data)
    pivot_mean = data.pivot(index='width_factor', columns='aggregator', values='mean_test_acc')
    pivot_std = data.pivot(index='width_factor', columns='aggregator', values='std_test_acc')
    
    ax = pivot_mean.plot(kind='bar', yerr=pivot_std, capsize=4, rot=0, figsize=(8,6), color=['#66c2a5', '#fc8d62'])
    
    plt.title("Defense Efficacy: Model Width vs Aggregation", fontsize=16, weight='bold')
    plt.xlabel("Model Width Factor", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.legend(title="Aggregator")
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp2_defense_comparison.png", dpi=300)
    plt.close()

def plot_exp3_mechanism(df):
    """
    Exp 3: Batch Size & Ordering Effect
    Heatmap หรือ Grouped Bar
    """
    print("Generating Plot: Exp 3 Mechanism...")
    data = df[df['phase'].str.contains("exp3", case=False, na=False)].copy()
    if data.empty: return

    # สร้างคอลัมน์ Group ใหม่เพื่อใช้เป็นแกน X
    data['Condition'] = "Batch=" + data['batch_size'].astype(str)
    
    plt.figure(figsize=(8, 6))
    
    sns.barplot(
        data=data,
        x="Condition",
        y="mean_test_acc",
        hue="data_ordering",
        palette="magma",
        edgecolor="black"
    )
    # (หมายเหตุ: Error bar ในส่วนนี้อาจวาด manual ยาก ข้ามไปก่อนได้ หรือใช้เทคนิค pivot เหมือน exp2)

    plt.title("Mechanism Analysis: Batch Size & Ordering", fontsize=16)
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp3_mechanism.png", dpi=300)
    plt.close()

def generate_latex_table(df):
    """
    สร้างตาราง LaTeX สำหรับใส่ใน Paper ได้เลย
    """
    print("Generating LaTeX Table...")
    output_path = f"{OUTPUT_DIR}/results_table.tex"
    
    with open(output_path, "w") as f:
        f.write("% Copy this into your LaTeX file\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Experiment & Config & Mean Acc & Std Dev \\\\\n\\hline\n")
        
        for index, row in df.iterrows():
            # จัด Format ชื่อ Config ให้สั้นลง
            config_str = f"W={row['width_factor']}"
            if 'poison_ratio' in row: config_str += f", P={row['poison_ratio']}"
            if 'aggregator' in row: config_str += f", {row['aggregator']}"
            
            f.write(f"{row['phase']} & {config_str} & {row['mean_test_acc']:.4f} & $\\pm$ {row['std_test_acc']:.4f} \\\\\n")
            
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Summary of Experimental Results (Mean $\\pm$ Std Dev over seeds)}\n")
        f.write("\\label{tab:main_results}\n\\end{table}\n")
        
    print(f"LaTeX table saved to {output_path}")
    
def plot_width_vs_depth(df):  # Accept df as parameter
    plt.figure(figsize=(8, 6))
    
    # 1. Plot เส้น Vary Width
    width_data = df[df['phase'].str.contains("width")]
    sns.lineplot(
        data=width_data, x="num_params", y="test_acc", 
        label="Scaling Width (Fixed Depth=4)", marker='o', color='blue'
    )
    
    # 2. Plot เส้น Vary Depth
    depth_data = df[df['phase'].str.contains("depth")]
    sns.lineplot(
        data=depth_data, x="num_params", y="test_acc", 
        label="Scaling Depth (Fixed Width=8)", marker='x', color='red'
    )
    
    plt.xscale('log') # แกน X เป็น Log Scale เพราะ Parameter จะเพิ่มทวีคูณ
    plt.title("Impact of Scaling Strategy: Width vs Depth")
    plt.xlabel("Number of Parameters (Log Scale)")
    plt.ylabel("Test Accuracy")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.savefig("./results_prelim/width_vs_depth.png")
    plt.show()

def main():
    ensure_dir(OUTPUT_DIR)
    df = load_data()
    if df is None: return

    # Plot ตาม Phase ใหม่
    plot_width_vs_depth(df)  # Pass df
    plot_exp1_double_descent(df)
    plot_exp2_defense(df)
    plot_exp3_mechanism(df)
    
    # สร้างตาราง LaTeX
    generate_latex_table(df)
    
    print("\nDefinitive Analysis Complete! Check ./results_definitive/plots folder.")

if __name__ == "__main__":
    main()