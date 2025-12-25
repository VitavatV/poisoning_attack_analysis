"""Simple verification of poisoning CSV"""
import pandas as pd

csv_path = 'poisoning_data/exp1_cifar10_config002_seed42_alpha100.0_poison0.3_label_flip.csv'
df = pd.read_csv(csv_path)

print(f"Total: {len(df)}, Poisoned: {df['Is_Poisoned'].sum()}, Pct: {100*df['Is_Poisoned'].sum()/len(df):.1f}%")

# Check label flipping for poisoned samples
poisoned = df[df['Is_Poisoned'] == True]
print("\nLabel flip check (first 5 poisoned samples):")
print(poisoned[['GT', 'Training_Class']].head())
print(f"\nExpected pattern: GT 0->9, 1->8, 2->7, 3->6, 4->5, etc.")

# Check client distribution
clients = df.groupby('Training_Client')['Is_Poisoned'].agg(['sum', 'count'])
clients['is_malicious'] = clients['sum'] > 0
print(f"\nMalicious clients: {clients['is_malicious'].sum()}/10")
print(f"Expected: 3 clients malicious (30% of 10 clients)")
