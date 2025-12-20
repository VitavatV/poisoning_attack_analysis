"""
Verify that experiment manager generates correct experiment counts
"""
import yaml
import glob
import itertools

def count_experiments_from_config(config_path):
    """Count experiments for a single config file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    phase_names = [
        'exp1_vary_width',
        'exp4_iid_vs_noniid',
        'exp5_defense_comparison',
        'exp2_mechanism_analysis',
        'exp3_attack_types',
    ]
    
    counts = {}
    
    for phase_name in phase_names:
        if phase_name not in config:
            continue
        
        phase_cfg = config[phase_name]
        vary_params = {}
        
        for item in phase_cfg['combinations']:
            vary_params.update(item)
        
        # Calculate total combinations
        vals = list(vary_params.values())
        total = 1
        for v in vals:
            total *= len(v)
        
        counts[phase_name] = total
    
    return counts

def main():
    config_files = sorted(glob.glob('configs/config_exp*.yaml'))
    
    all_counts = {}
    
    for config_path in config_files:
        print(f"\n{'='*60}")
        print(f"Config: {config_path}")
        print(f"{'='*60}")
        
        counts = count_experiments_from_config(config_path)
        
        for phase, count in counts.items():
            print(f"{phase}: {count} experiments")
            if phase not in all_counts:
                all_counts[phase] = 0
            all_counts[phase] += count
    
    print(f"\n{'='*60}")
    print("TOTAL EXPERIMENT COUNTS")
    print(f"{'='*60}")
    
    total = 0
    for phase, count in sorted(all_counts.items()):
        print(f"{phase}: {count} experiments")
        total += count
    
    print(f"\nGRAND TOTAL: {total} experiments")
    
    print(f"\n{'='*60}")
    print("MANUAL CALCULATIONS VERIFICATION")
    print(f"{'='*60}")
    
    expected = {
        'exp1_vary_width': 960,
        'exp2_mechanism_analysis': 540,
        'exp3_attack_types': 120,
        'exp4_iid_vs_noniid': 240,
        'exp5_defense_comparison': 120
    }
    
    all_match = True
    for phase, expected_count in expected.items():
        actual_count = all_counts.get(phase, 0)
        match = "✓" if actual_count == expected_count else "✗"
        print(f"{phase}: Expected {expected_count}, Got {actual_count} {match}")
        if actual_count != expected_count:
            all_match = False
    
    expected_total = sum(expected.values())
    print(f"\nTOTAL: Expected {expected_total}, Got {total} {'✓' if total == expected_total else '✗'}")
    
    if all_match:
        print("\n🎉 All experiment counts match the manual calculations!")
    else:
        print("\n⚠️ Some experiment counts don't match!")

if __name__ == "__main__":
    main()
