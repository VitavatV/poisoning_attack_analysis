"""
Verify final_results.csv structure matches code expectations
"""
import pandas as pd

# Expected columns based on run_task() in experiment_runner_gpu.py
expected_columns = [
    # Signature parameters (12)
    "phase",
    "dataset", 
    "model_type",
    "width_factor",
    "depth",
    "poison_ratio",
    "poison_type",
    "alpha",
    "data_ordering",
    "aggregator",
    "batch_size",
    # Results (10)
    "mean_test_acc",
    "std_test_acc",
    "mean_test_loss",
    "std_test_loss",
    "mean_val_acc",
    "std_val_acc",
    "mean_val_loss",
    "std_val_loss",
    "num_parameters",
    "best_epoch",
    "seed"
]

print("="*60)
print("CSV STRUCTURE VERIFICATION")
print("="*60)

# Read actual CSV
try:
    df = pd.read_csv('final_results.csv')
    actual_columns = df.columns.tolist()
    
    print(f"\nTotal rows in final_results.csv: {len(df)}")
    print(f"\nActual columns ({len(actual_columns)}):")
    for i, col in enumerate(actual_columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nExpected columns ({len(expected_columns)}):")
    for i, col in enumerate(expected_columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Check for differences
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    missing = set(expected_columns) - set(actual_columns)
    extra = set(actual_columns) - set(expected_columns)
    
    if not missing and not extra:
        print("\n✅ PERFECT MATCH! All columns match exactly.")
    else:
        if missing:
            print(f"\n❌ Missing columns ({len(missing)}):")
            for col in missing:
                print(f"  - {col}")
        
        if extra:
            print(f"\n⚠️ Extra columns ({len(extra)}):")
            for col in extra:
                print(f"  - {col}")
    
    # Sample data
    print("\n" + "="*60)
    print("SAMPLE DATA (first row)")
    print("="*60)
    if len(df) > 0:
        first_row = df.iloc[0]
        for col in actual_columns:
            print(f"{col:20s}: {first_row[col]}")
    else:
        print("No data in CSV")

except FileNotFoundError:
    print("\n❌ final_results.csv not found!")
except Exception as e:
    print(f"\n❌ Error reading CSV: {e}")
