"""
Test script to verify IEEE figure generation works correctly
Runs a simple test to ensure the figure_utils module works as expected
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# Import the figure utility
try:
    from figure_utils import save_ieee_figure
    print("✓ Successfully imported figure_utils module")
except ImportError as e:
    print(f"❌ Failed to import figure_utils: {e}")
    sys.exit(1)


def test_figure_generation():
    """Test IEEE figure generation with a simple plot."""
    
    print("\nTesting IEEE figure generation...")
    print("=" * 60)
    
    # Create a simple test figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate some test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Test Figure for IEEE Format Verification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save using IEEE format
    print("\nGenerating test figures...")
    saved_files = save_ieee_figure('test_figure', 'exp0', 'test', '01', fig)
    
    # Verify files were created
    print("\nVerifying generated files...")
    all_exist = True
    
    for filepath in saved_files:
        path = Path(filepath)
        if path.exists():
            # Check file size and DPI
            with Image.open(path) as img:
                dpi = img.info.get('dpi', (0, 0))
                size_inches = (img.width / dpi[0], img.height / dpi[1]) if dpi[0] > 0 else (0, 0)
                
                print(f"\n  File: {path.name}")
                print(f"    ✓ Exists")
                print(f"    ✓ Resolution: {img.width}x{img.height} pixels")
                print(f"    ✓ DPI: {dpi[0]}")
                print(f"    ✓ Size: {size_inches[0]:.2f}\" x {size_inches[1]:.2f}\"")
                
                # Verify DPI is 600
                if dpi[0] != 600:
                    print(f"    ⚠ WARNING: Expected 600 DPI, got {dpi[0]}")
                    all_exist = False
                
                # Verify width matches expected (3.5" or 7.16")
                expected_widths = [3.5, 7.16]
                width_match = any(abs(size_inches[0] - w) < 0.1 for w in expected_widths)
                if not width_match:
                    print(f"    ⚠ WARNING: Width {size_inches[0]:.2f}\" doesn't match 3.5\" or 7.16\"")
                    all_exist = False
        else:
            print(f"  ❌ File not found: {filepath}")
            all_exist = False
    
    plt.close(fig)
    
    return all_exist


def check_notebook_modifications():
    """Check if all notebooks were modified correctly."""
    
    print("\n" + "=" * 60)
    print("Checking notebook modifications...")
    print("=" * 60)
    
    base_path = Path('.')
    notebooks = sorted(base_path.glob('analysis_exp*.ipynb'))
    
    if not notebooks:
        print("❌ No analysis notebooks found!")
        return False
    
    print(f"\nFound {len(notebooks)} notebooks")
    
    all_modified = True
    for notebook_path in notebooks:
        # Read notebook and check for required imports and calls
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        has_import = 'from figure_utils import save_ieee_figure' in content
        has_calls = 'save_ieee_figure(' in content
        
        status = "✓" if (has_import and has_calls) else "❌"
        print(f"  {status} {notebook_path.name}")
        
        if not (has_import and has_calls):
            all_modified = False
            if not has_import:
                print(f"      Missing import statement")
            if not has_calls:
                print(f"      Missing save_ieee_figure() calls")
    
    return all_modified


def main():
    """Run all verification tests."""
    
    print("\n" + "=" * 60)
    print(" IEEE Figure Format Verification Test")
    print("=" * 60)
    
    # Test 1: Figure generation
    test1_pass = test_figure_generation()
    
    # Test 2: Notebook modifications
    test2_pass = check_notebook_modifications()
    
    # Summary
    print("\n" + "=" * 60)
    print(" Test Summary")
    print("=" * 60)
    print(f"  Figure Generation: {'✓ PASS' if test1_pass else '❌ FAIL'}")
    print(f"  Notebook Modifications: {'✓ PASS' if test2_pass else '❌ FAIL'}")
    print("=" * 60)
    
    if test1_pass and test2_pass:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("  1. Run your analysis notebooks to generate all figures")
        print("  2. Check the 'figure/' directory for IEEE format outputs")
        print("  3. Open the PNG files to verify quality and sizing")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    exit(main())
