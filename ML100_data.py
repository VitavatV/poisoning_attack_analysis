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
N_POISONED_CLIENTS = config.get("N_POISONED_CLIENTS", 0)
ORIGINAL_M = config.get("ORIGINAL_M", 0.7)
ORIGINAL_B = config.get("ORIGINAL_B", 0.1)
ORIGINAL_NOISE_SPACE = config.get("ORIGINAL_NOISE_SPACE", 0.1)
ORIGINAL_NOISE_RANGE = config.get("ORIGINAL_NOISE_RANGE", 0.9)

# Set random seeds for reproducibility
np.random.seed(0)

NAME_SAVE_PATH = "data"
SAVE_PATH = os.path.join("results",NAME_SAVE_PATH)
# Remove existing directory if it exists
if os.path.exists(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)
os.makedirs(SAVE_PATH, exist_ok=True)

def save_client_data(client_id, is_poisoned=False):

    # 1. Generate static data
    x_class0 = np.random.uniform(0.0, 1.0, N_PER_CLASS)
    y_bias = np.random.uniform(ORIGINAL_NOISE_SPACE, ORIGINAL_NOISE_RANGE, N_PER_CLASS)
    y_class0 = ORIGINAL_M * x_class0 + ORIGINAL_B - y_bias
    x_class1 = np.random.uniform(0.0, 1.0, N_PER_CLASS)
    y_bias = np.random.uniform(ORIGINAL_NOISE_SPACE, ORIGINAL_NOISE_RANGE, N_PER_CLASS)
    y_class1 = ORIGINAL_M * x_class1 + ORIGINAL_B + y_bias

    X = np.concatenate([np.stack([x_class0, y_class0], axis=1), np.stack([x_class1, y_class1], axis=1)], axis=0)

    if not is_poisoned:
        # 2. save non-poisoned data
        y = np.array([0]*N_PER_CLASS + [1]*N_PER_CLASS)
        
        # Save X and y for next use
        np.save(os.path.join(SAVE_PATH, f"ML100_data_X_client{client_id}.npy"), X)
        np.save(os.path.join(SAVE_PATH, f"ML100_data_y_client{client_id}.npy"), y)

    else:
        # 3. save poisoned data
        y = np.array([1]*N_PER_CLASS + [0]*N_PER_CLASS)
        

        # Save X and y for next use
        np.save(os.path.join(SAVE_PATH, f"ML100_poison_X_client{client_id}.npy"), X)
        np.save(os.path.join(SAVE_PATH, f"ML100_poison_y_client{client_id}.npy"), y)

for i in range(N_CLIENTS):
    save_client_data(i, is_poisoned=False)
for i in range(N_POISONED_CLIENTS):
    save_client_data(i, is_poisoned=True)