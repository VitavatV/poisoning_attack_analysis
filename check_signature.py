"""
Verify signature column was added
"""
import pandas as pd
import time

# Wait a moment for any file locks to clear
time.sleep(1)

try:
    df = pd.read_csv('final_results.csv')
    print("Column order:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Has signature column: {'signature' in df.columns}")
    
    if 'signature' in df.columns:
        print(f"\nSample signature (first row):")
        print(f"  {df['signature'].iloc[0]}")
    
except Exception as e:
    print(f"Error: {e}")
