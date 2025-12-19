"""Verify experiment number swap"""
import yaml
import os

# Expected mapping: 2→5, 3→2, 4→3, 1→4, 0→1
mapping = {
    'exp1': 'exp0',  # exp0 became exp1
    'exp2': 'exp3',  # exp3 became exp2
    'exp3': 'exp4',  # exp4 became exp3
    'exp4': 'exp1',  # exp1 became exp4
    'exp5': 'exp2',  # exp2 became exp5
}

print("="*60)
print("Verifying Experiment Number Swap")
print("="*60)
print("\nMapping: exp0→exp1, exp1→exp4, exp2→exp5, exp3→exp2, exp4→exp3\n")

configs_dir = 'c:/github/poisoning_attack_analysis/configs'
all_pass = True

for new_exp, old_exp in mapping.items():
    config_file = os.path.join(configs_dir, f'config_{new_exp}.yaml')
    
    print(f"\nChecking {new_exp} (was {old_exp}):")
    
    with open(config_file, 'r') as f:
        content = f.read()
        config = yaml.safe_load(content)
    
    # Check header
    if content.startswith(f'# EXP {new_exp[-1]}'):
        print(f"  ✅ Header: EXP {new_exp[-1]}")
    else:
        print(f"  ❌ Header incorrect")
        all_pass = False
    
    # Check project name
    project_name = config['defaults']['project_name']
    if f'EXP{new_exp[-1]}' in project_name:
        print(f"  ✅ Project name: {project_name}")
    else:
        print(f"  ❌ Project name: {project_name}")
        all_pass = False
    
    # Check output_dir
    output_dir = config['defaults']['output_dir']
    if f'results_{new_exp}' in output_dir:
        print(f"  ✅ Output dir: {output_dir}")
    else:
        print(f"  ❌ Output dir: {output_dir}")
        all_pass = False
    
    # Check phase name
    phase_keys = [k for k in config.keys() if k.startswith('exp')]
    if phase_keys:
        phase_name = phase_keys[0]
        if phase_name.startswith(new_exp):
            print(f"  ✅ Phase name: {phase_name}")
        else:
            print(f"  ❌ Phase name: {phase_name}")
            all_pass = False

print("\n" + "="*60)
if all_pass:
    print("✅ All verifications PASSED!")
else:
    print("❌ Some verifications FAILED!")
print("="*60)
