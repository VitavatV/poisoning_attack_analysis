"""
Script to generate CIFAR10 analysis notebooks from MNIST templates
"""
import json
import re

def create_cifar10_notebook(mnist_file, cifar10_file, exp_num):
    """Convert MNIST notebook to CIFAR10 version"""
    
    with open(mnist_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Replacements to make
    replacements = {
        '(MNIST)': '(CIFAR10)',
        'MNIST': 'CIFAR10',
        './results_exp{}_mnist'.format(exp_num): './results_exp{}_cifar10'.format(exp_num),
        'results_exp{}_mnist'.format(exp_num): 'results_exp{}_cifar10'.format(exp_num),
    }
    
    # Apply replacements to all cells
    for cell in notebook['cells']:
        if 'source' in cell:
            if isinstance(cell['source'], list):
                for i in range(len(cell['source'])):
                    for old, new in replacements.items():
                        cell['source'][i] = cell['source'][i].replace(old, new)
            elif isinstance(cell['source'], str):
                for old, new in replacements.items():
                    cell['source'] = cell['source'].replace(old, new)
    
    # Save CIFAR10 version
    with open(cifar10_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✓ Created {cifar10_file}")

# Create all CIFAR10 notebooks
experiments = [0, 1, 2, 3, 4]

for exp_num in experiments:
    mnist_file = f'analysis_exp{exp_num}_mnist.ipynb'
    cifar10_file = f'analysis_exp{exp_num}_cifar10.ipynb'
    
    try:
        create_cifar10_notebook(mnist_file, cifar10_file, exp_num)
    except FileNotFoundError:
        print(f"❌ Error: {mnist_file} not found")
    except Exception as e:
        print(f"❌ Error creating {cifar10_file}: {e}")

print("\\n✓ All CIFAR10 notebooks created successfully!")
