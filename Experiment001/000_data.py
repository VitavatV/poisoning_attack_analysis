import os
import numpy as np
import json
import shutil

config = json.load(open("config.json"))
LEARNING_RATE = config.get("LEARNING_RATE", 0.5)
EPOCHS = config.get("EPOCHS", 32)
N_PER_CLASS = config.get("N_PER_CLASS", 32)
N_CLASSES = config.get("N_CLASSES", 2)
N_CLIENTS = config.get("N_CLIENTS", 2)
N_TEST = config.get("N_TEST", 1000)
N_POISONED_CLIENTS = config.get("N_POISONED_CLIENTS", 0)
ORIGINAL_M = config.get("ORIGINAL_M", 0.7)
ORIGINAL_B = config.get("ORIGINAL_B", 0.1)
ORIGINAL_NOISE_SPACE = config.get("ORIGINAL_NOISE_SPACE", 0.1)
ORIGINAL_NOISE_RANGE = config.get("ORIGINAL_NOISE_RANGE", 0.9)

# Set random seeds for reproducibility
np.random.seed(0)

def random_distribution(mode='uniform',low=0.0, high=1.0, size=10):
    if mode == 'uniform':
        return np.random.uniform(low, high, size)
    elif mode == 'normal':
        return np.random.normal((low+high)/2, (high-low)/6, size)
    elif mode == 'laplace':
        return np.random.laplace((low+high)/2, (high-low)/6, size)
    elif mode == 'beta':
        return np.random.beta(2, 5, size) * (high - low) + low
    elif mode == 'triangular':
        return np.random.triangular(low, (low+high)/2, high, size)
    else:
        raise ValueError("Unsupported distribution mode")

NAME_SAVE_PATH = "data"
SAVE_PATH = os.path.join("results",NAME_SAVE_PATH)
# Remove existing directory if it exists
if os.path.exists(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)
os.makedirs(SAVE_PATH, exist_ok=True)

def save_client_data(local_save_path = '', client_id = 0, is_poisoned=False, is_test=False, mode='uniform'):
    local_N_PER_CLASS = N_PER_CLASS

    if is_test:
        local_N_PER_CLASS = N_TEST // N_CLASSES

    # 1. Generate static data
    x_class0 = random_distribution(mode, 0.0, 1.0, local_N_PER_CLASS)
    y_bias = random_distribution(mode, ORIGINAL_NOISE_SPACE, ORIGINAL_NOISE_RANGE, local_N_PER_CLASS)
    y_class0 = ORIGINAL_M * x_class0 + ORIGINAL_B - y_bias
    x_class1 = random_distribution(mode, 0.0, 1.0, local_N_PER_CLASS)
    y_bias = random_distribution(mode, ORIGINAL_NOISE_SPACE, ORIGINAL_NOISE_RANGE, local_N_PER_CLASS)
    y_class1 = ORIGINAL_M * x_class1 + ORIGINAL_B + y_bias

    X = np.concatenate([np.stack([x_class0, y_class0], axis=1), np.stack([x_class1, y_class1], axis=1)], axis=0)

    if not is_poisoned:
        # 2. save non-poisoned data
        y = np.array([0]*local_N_PER_CLASS + [1]*local_N_PER_CLASS)
        
        if is_test:
            # Save X and y for next use
            np.save(os.path.join(local_save_path, f"ML100_test_X_client.npy"), X)
            np.save(os.path.join(local_save_path, f"ML100_test_y_client.npy"), y)
        else:
            # Save X and y for next use
            np.save(os.path.join(local_save_path, f"ML100_data_X_client{client_id}.npy"), X)
            np.save(os.path.join(local_save_path, f"ML100_data_y_client{client_id}.npy"), y)

    else:
        # 3. save poisoned data
        y = np.array([1]*local_N_PER_CLASS + [0]*local_N_PER_CLASS)
        

        # Save X and y for next use
        np.save(os.path.join(local_save_path, f"ML100_poison_X_client{client_id}.npy"), X)
        np.save(os.path.join(local_save_path, f"ML100_poison_y_client{client_id}.npy"), y)

for mode in ['uniform', 'normal', 'laplace', 'beta', 'triangular']:
    local_save_path = os.path.join(SAVE_PATH, mode)
    os.makedirs(local_save_path, exist_ok=True)
    for i in range(N_CLIENTS):
        save_client_data(local_save_path, i, is_poisoned=False, is_test=False, mode=mode)
    for i in range(N_POISONED_CLIENTS):
        save_client_data(local_save_path, i, is_poisoned=True, is_test=False, mode=mode)

    save_client_data(local_save_path, 0, is_poisoned=False, is_test=True, mode=mode)