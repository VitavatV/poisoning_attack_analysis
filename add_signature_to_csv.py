"""
Add signature column to existing final_results.csv files
"""
import pandas as pd
import glob
import os

def create_signature(row):
    """Create signature from row data"""
    return f"{row['dataset']}|{row['model_type']}|{row['width_factor']}|{row['depth']}|{row['poison_ratio']}|{row['poison_type']}|{row['alpha']}|{row['data_ordering']}|{row['aggregator']}|{row['batch_size']}|{row['seed']}"

def update_csv_with_signature(csv_path):
    """Add signature column to CSV if it doesn't exist"""
    try:
        df = pd.read_csv(csv_path)
        
        # Check if signature column already exists
        if 'signature' in df.columns:
            print(f"✓ {csv_path} already has signature column")
            return True
        
        # Create signature column
        df['signature'] = df.apply(create_signature, axis=1)
        
        # Reorder columns to put signature after phase
        cols = df.columns.tolist()
        if 'phase' in cols:
            # Move signature to right after phase
            cols.remove('signature')
            phase_idx = cols.index('phase')
            cols.insert(phase_idx + 1, 'signature')
            df = df[cols]
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        print(f"✓ Updated {csv_path} with signature column ({len(df)} rows)")
        return True
        
    except Exception as e:
        print(f"✗ Error updating {csv_path}: {e}")
        return False

def main():
    print("="*60)
    print("ADDING SIGNATURE COLUMN TO CSV FILES")
    print("="*60)
    
    # Find all final_results.csv files
    csv_files = []
    
    # Root level
    if os.path.exists('final_results.csv'):
        csv_files.append('final_results.csv')
    
    # Results directories
    result_dirs = glob.glob('results_*')
    for result_dir in result_dirs:
        csv_path = os.path.join(result_dir, 'final_results.csv')
        if os.path.exists(csv_path):
            csv_files.append(csv_path)
    
    print(f"\nFound {len(csv_files)} CSV file(s) to update:\n")
    
    success_count = 0
    for csv_file in csv_files:
        if update_csv_with_signature(csv_file):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Updated {success_count}/{len(csv_files)} files successfully")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
