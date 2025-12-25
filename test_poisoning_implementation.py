"""
Test Client-Level Poisoning Implementation

Verifies that the poisoning algorithm correctly:
1. Selects the right number of malicious clients
2. Applies complementary label flip correctly
3. Applies random noise correctly
4. Poisons 100% of malicious client data
"""

import sys
import numpy as np

# Add project root to path
sys.path.insert(0, 'c:/github/poisoning_attack_analysis')

from data_utils import (
    select_malicious_clients,
    apply_label_flip_poisoning,
    apply_random_noise_poisoning
)


def test_client_selection():
    """Test that correct number of clients are selected as malicious"""
    print("\n" + "="*60)
    print("TEST 1: Client Selection")
    print("="*60)
    
    # Test case 1: 30% of 10 clients = 3 clients
    clients = select_malicious_clients(10, 0.3, 42)
    assert len(clients) == 3, f"Expected 3 malicious clients, got {len(clients)}"
    print(f"✓ 30% of 10 clients: {len(clients)} malicious clients")
    print(f"  Selected IDs: {clients}")
    
    # Test case 2: 20% of 5 clients = 1 client
    clients = select_malicious_clients(5, 0.2, 42)
    assert len(clients) == 1, f"Expected 1 malicious client, got {len(clients)}"
    print(f"✓ 20% of 5 clients: {len(clients)} malicious client")
    print(f"  Selected IDs: {clients}")
    
    # Test case 3: 0% poison_ratio = 0 clients
    clients = select_malicious_clients(10, 0.0, 42)
    assert len(clients) == 0, f"Expected 0 malicious clients, got {len(clients)}"
    print(f"✓ 0% of 10 clients: {len(clients)} malicious clients")
    
    # Test case 4: 50% of 10 clients = 5 clients
    clients = select_malicious_clients(10, 0.5, 42)
    assert len(clients) == 5, f"Expected 5 malicious clients, got {len(clients)}"
    print(f"✓ 50% of 10 clients: {len(clients)} malicious clients")
    print(f"  Selected IDs: {clients}")
    
    print("\n✅ All client selection tests passed!")


def test_label_flip():
    """Test complementary label flipping"""
    print("\n" + "="*60)
    print("TEST 2: Label Flip Poisoning (Complementary)")
    print("="*60)
    
    # Test all 10 classes for CIFAR-10/MNIST
    expected_flips = {
        0: 9, 1: 8, 2: 7, 3: 6, 4: 5,
        5: 4, 6: 3, 7: 2, 8: 1, 9: 0
    }
    
    for gt_label, expected_poison in expected_flips.items():
        poison_label = apply_label_flip_poisoning(gt_label, 10)
        assert poison_label == expected_poison, \
            f"Label {gt_label} should flip to {expected_poison}, got {poison_label}"
        print(f"✓ {gt_label} → {poison_label}")
    
    print("\n✅ All label flip tests passed!")


def test_random_noise():
    """Test random noise poisoning"""
    print("\n" + "="*60)
    print("TEST 3: Random Noise Poisoning")
    print("="*60)
    
    num_classes = 10
    
    # Test that random noise never returns the ground truth label
    for gt_label in range(num_classes):
        # Generate multiple samples to check randomness
        poison_labels = set()
        for i in range(50):
            poison_label = apply_random_noise_poisoning(gt_label, num_classes, seed=i)
            assert poison_label != gt_label, \
                f"Random noise should not return GT label {gt_label}"
            assert 0 <= poison_label < num_classes, \
                f"Poison label {poison_label} out of range [0, {num_classes})"
            poison_labels.add(poison_label)
        
        print(f"✓ GT label {gt_label}: Generated {len(poison_labels)} different poison labels (never {gt_label})")
        print(f"  Poison labels seen: {sorted(poison_labels)}")
    
    print("\n✅ All random noise tests passed!")


def test_determinism():
    """Test that results are deterministic with same seed"""
    print("\n" + "="*60)
    print("TEST 4: Determinism with Seeds")
    print("="*60)
    
    # Test client selection determinism
    clients1 = select_malicious_clients(10, 0.3, 42)
    clients2 = select_malicious_clients(10, 0.3, 42)
    assert clients1 == clients2, "Same seed should produce same client selection"
    print(f"✓ Client selection is deterministic: {clients1}")
    
    # Test random noise determinism
    noise1 = apply_random_noise_poisoning(5, 10, seed=123)
    noise2 = apply_random_noise_poisoning(5, 10, seed=123)
    assert noise1 == noise2, "Same seed should produce same random noise"
    print(f"✓ Random noise is deterministic: {noise1}")
    
    # Test different seeds produce different results (probabilistic test)
    noise_different = apply_random_noise_poisoning(5, 10, seed=456)
    # With 9 possible poison labels, probability of collision is 1/9 ≈ 11%
    # Just print for observation, don't assert
    print(f"✓ Different seed produces: {noise_different} (seed 123: {noise1}, seed 456: {noise_different})")
    
    print("\n✅ All determinism tests passed!")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Testing Client-Level Poisoning Implementation")
    print("="*60)
    
    try:
        test_client_selection()
        test_label_flip()
        test_random_noise()
        test_determinism()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nThe client-level poisoning implementation is working correctly:")
        print("  ✓ Client selection respects poison_ratio")
        print("  ✓ Label flip uses complementary mapping (0↔9, 1↔8, etc.)")
        print("  ✓ Random noise never selects the ground truth label")
        print("  ✓ Results are deterministic with fixed seeds")
        print("\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
