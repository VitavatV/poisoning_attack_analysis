"""
Script to modify all analysis notebooks to use IEEE figure format
Replaces plt.savefig() calls with save_ieee_figure() calls
"""
import json
import re
from pathlib import Path


def modify_notebook(notebook_path):
    """Modify a single notebook to use IEEE figure format."""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Extract experiment name and dataset from filename
    # Format: analysis_exp{N}_{dataset}.ipynb
    filename = notebook_path.stem
    match = re.match(r'analysis_(exp\d+)_(\w+)', filename)
    if not match:
        print(f"⚠ Skipping {filename} - doesn't match expected pattern")
        return False
    
    exp_name, dataset = match.groups()
    
    # Track figure number
    figure_num = 1
    modified = False
    
    # Iterate through cells
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, str):
                source = [source]
            
            new_source = []
            import_added = False
            
            for i, line in enumerate(source):
                # Add import at the beginning of first code cell if not already there
                if i == 0 and not import_added and figure_num == 1:
                    if 'from figure_utils import' not in ''.join(source):
                        new_source.append('from figure_utils import save_ieee_figure\n')
                    import_added = True
                
                # Check if this line contains plt.savefig
                if 'plt.savefig' in line:
                    # Extract the output directory and filename from the savefig call
                    # Format: plt.savefig(f"{OUTPUT_DIR}/filename.png")
                    match_savefig = re.search(r'plt\.savefig\(f?["\'].*?/(\w+)\.png["\'].*?\)', line)
                    
                    if match_savefig:
                        figure_name = match_savefig.group(1)
                        
                        # Generate figure number with leading zero
                        fig_num_str = f"{figure_num:02d}"
                        
                        # Replace the plt.savefig line with save_ieee_figure call
                        indent = len(line) - len(line.lstrip())
                        indent_str = ' ' * indent
                        
                        replacement = f'{indent_str}save_ieee_figure("{figure_name}", "{exp_name}", "{dataset}", "{fig_num_str}")\n'
                        new_source.append(replacement)
                        
                        figure_num += 1
                        modified = True
                        continue
                
                new_source.append(line)
            
            cell['source'] = new_source
    
    if modified:
        # Save modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"✓ Modified {filename}: {figure_num-1} figures updated")
        return True
    else:
        print(f"  No changes needed for {filename}")
        return False


def main():
    """Process all analysis notebooks."""
    base_path = Path('c:/github/poisoning_attack_analysis')
    
    # Find all analysis notebooks
    notebooks = sorted(base_path.glob('analysis_exp*.ipynb'))
    
    if not notebooks:
        print("❌ No analysis notebooks found!")
        return
    
    print(f"Found {len(notebooks)} notebooks to process\n")
    
    modified_count = 0
    for notebook_path in notebooks:
        if modify_notebook(notebook_path):
            modified_count += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully modified {modified_count}/{len(notebooks)} notebooks")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Run the notebooks to generate IEEE format figures")
    print("2. Check the 'figure/' directory for output")
    print("3. Verify figures are 600 DPI with 3.5\" and 7.16\" widths")


if __name__ == '__main__':
    main()
