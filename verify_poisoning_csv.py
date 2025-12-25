"""
Verify poisoning data CSV files
"""
import pandas as pd
import numpy as np

# Load a sample CSV with poisoning
csv_path = 'poisoning_data/exp1_cifar10_config002_seed42_alpha100.0_poison0.3_label_flip.csv'
df = pd.read_csv(csv_path)

print("="*80)
print("POISONING DATA VERIFICATION")
print("="*80)
print(f"\nFile: {csv_path}")
print(f"\nTotal samples: {len(df)}")
print(f"Poisoned samples: {df['Is_Poisoned'].sum()}")
print(f"Poison percentage: {100*df['Is_Poisoned'].sum()/len(df):.1f}%")

# Show sample rows
print("\n" + "="*80)
print("SAMPLE ROWS (showing poisoned samples)")
print("="*80)
poisoned_samples = df[df['Is_Poisoned'] == True].head(10)
print(poisoned_samples[['Sample_Index', 'GT', 'Training_Class', 'Training_Client', 'Is_Poisoned']])

# Verify label flip pattern (should be complementary: 0→9, 1→8, etc.)
print("\n" + "="*80)
print("LABEL FLIP VERIFICATION (should be complementary)")
print("="*80)
poisoned_df = df[df['Is_Poisoned'] == True]
for gt_class in range(10):
    samples = poisoned_df[poisoned_df['GT'] == gt_class]
    if len(samples) > 0:
        training_classes = samples['Training_Class'].unique()
        expected_flip = 9 - gt_class
        print(f"GT class {gt_class} → Training class {training_classes} (Expected: {expected_flip})")
        
# Show crosstab
print("\n" + "="*80)
print("CROSSTAB: GT vs Training_Class")
print("="*80)
ct = pd.crosstab(df['GT'], df['Training_Class'])
print(ct)

# Show client distribution
print("\n" + "="*80)
print("CLIENT DISTRIBUTION")
print("="*80)
client_stats = df.groupby('Training_Client').agg({
    'Sample_Index': 'count',
    'Is_Poisoned': 'sum'
}).rename(columns={'Sample_Index': 'Total_Samples', 'Is_Poisoned': 'Poisoned_Samples'})
client_stats['Is_Malicious'] = client_stats['Poisoned_Samples'] > 0
client_stats['Poison_Pct'] = 100 * client_stats['Poisoned_Samples'] / client_stats['Total_Samples']
print(client_stats)

malicious_count = client_stats['Is_Malicious'].sum()
print(f"\nMalicious clients: {malicious_count}/10 (30% expected)")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
