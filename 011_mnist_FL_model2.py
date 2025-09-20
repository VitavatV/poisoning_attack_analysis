import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# class MNISTNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(28*28, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 1)
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = torch.sigmoid(self.fc2(x))
#         return x

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# class MNISTNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(28*28, 1)
#     def forward(self, x):
#         x = torch.sigmoid(self.fc1(x))
#         return x

def get_client_data(data_dir, client_type, idx, img_size=28):
    X, y = [], []
    folder = os.path.join(data_dir, f"{client_type}_{idx}")
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if not os.path.isdir(label_folder):
            continue
        for fname in os.listdir(label_folder):
            if fname.endswith('.png') or fname.endswith('.jpg'):
                img = Image.open(os.path.join(label_folder, fname)).convert('L').resize((img_size, img_size))
                X.append(np.array(img).flatten() / 255.0)
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def average_weights(w_list):
    avg = {}
    for k in w_list[0].keys():
        avg[k] = sum([w[k] for w in w_list]) / len(w_list)
    return avg

def binary_cross_entropy(pred, target):
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1 - eps)
    return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()

# Load test data once
def load_mnist_binary_test_data_flat(test_dir, img_size=28):
    X = []
    y = []
    if not os.path.exists(test_dir):
        return X, y
    for label in ['0', '1']:
        label_folder = os.path.join(test_dir, label)
        if not os.path.isdir(label_folder):
            continue
        for fname in os.listdir(label_folder):
            if fname.endswith('.png') or fname.endswith('.jpg'):
                img = Image.open(os.path.join(label_folder, fname)).convert('L').resize((img_size, img_size))
                X.append(np.array(img).flatten() / 255.0)
                y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

config = json.load(open("config_mnist.json"))
LEARNING_RATE = config.get("LEARNING_RATE", 0.01)
START_EPOCH = config.get("START_EPOCH", 0)
EPOCHS = 1
ROUNDS = config.get("EPOCHS", 10)
BATCH_SIZE = config.get("BATCH_SIZE", 10)
N_HONEST = config.get("N_HONEST", 100)
N_POISONED = config.get("N_POISONED", 0)
N_POISONED_CLIENTS = config.get("N_POISONED_CLIENTS", 2)
IMG_SIZE = 28

np.random.seed(0)
torch.manual_seed(0)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_PATH = "./data/mnist_binary_poison/train"
TEST_PATH = "./data/mnist_images/test"

NAME_SAVE_PATH = f"FL_MODEL2_BATCH_SIZE{BATCH_SIZE}"
mode = "mnist_binary"
for i_poisoned in range(START_EPOCH,N_POISONED_CLIENTS+1):

    NAME_SAVE_update_PATH = f"poisoned_{i_poisoned}"
    SAVE_PATH = os.path.join("results",NAME_SAVE_PATH,mode,NAME_SAVE_update_PATH)
    # Remove existing directory if it exists
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    SAVE_all_PATH = os.path.join("results",NAME_SAVE_PATH,mode,'all')
    os.makedirs(SAVE_all_PATH, exist_ok=True)

    X_test, y_test = load_mnist_binary_test_data_flat(TEST_PATH, img_size=IMG_SIZE)
    if X_test.shape[0] == 0:
        raise ValueError("No valid test images found in the test directory!")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    global_model = MNISTNet().to(device)
    global_weights = global_model.state_dict()

    records = []

    X_train_tensor = {}
    y_train_tensor = {}
    local_models = {}
    local_optimizers = {}
    for i in range(N_HONEST):
        X_c, y_c = get_client_data(DATA_PATH, "honest", i, img_size=IMG_SIZE)
        if len(X_c) == 0:
            continue
        # model = MNISTNet().to(device)
        # model.load_state_dict(global_weights)
        # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        X_c_tensor = torch.tensor(X_c, dtype=torch.float32).to(device)
        y_c_tensor = torch.tensor(y_c, dtype=torch.float32).unsqueeze(1).to(device)
        X_train_tensor[f"honest_{i}"] = X_c_tensor
        y_train_tensor[f"honest_{i}"] = y_c_tensor
    for i in range(i_poisoned):
        X_c, y_c = get_client_data(DATA_PATH, "poison", i, img_size=IMG_SIZE)
        if len(X_c) == 0:
            continue
        # model = MNISTNet().to(device)
        # model.load_state_dict(global_weights)
        # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        X_c_tensor = torch.tensor(X_c, dtype=torch.float32).to(device)
        y_c_tensor = torch.tensor(y_c, dtype=torch.float32).unsqueeze(1).to(device)
        X_train_tensor[f"poison_{i}"] = X_c_tensor
        y_train_tensor[f"poison_{i}"] = y_c_tensor

    for round in range(ROUNDS):
        local_weights = []
        local_sizes = []

        # Each client trains locally
        for i in range(N_HONEST):
            X_c_tensor = X_train_tensor[f"honest_{i}"]
            y_c_tensor = y_train_tensor[f"honest_{i}"]
            model = MNISTNet().to(device)
            model.load_state_dict(global_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            model.train()
            for epoch in range(EPOCHS):
                for start in range(0, X_c_tensor.size(0), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    xb = X_c_tensor[start:end]
                    yb = y_c_tensor[start:end]
                    outputs = model(xb)
                    loss = binary_cross_entropy(outputs, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            local_weights.append(model.state_dict())
            local_sizes.append(len(X_c))

        for i in range(i_poisoned):
            X_c_tensor = X_train_tensor[f"poison_{i}"]
            y_c_tensor = y_train_tensor[f"poison_{i}"]
            model = MNISTNet().to(device)
            model.load_state_dict(global_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            model.train()
            for epoch in range(EPOCHS):
                for start in range(0, X_c_tensor.size(0), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    xb = X_c_tensor[start:end]
                    yb = y_c_tensor[start:end]
                    outputs = model(xb)
                    loss = binary_cross_entropy(outputs, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            local_weights.append(model.state_dict())
            local_sizes.append(len(X_c))

        # Federated averaging (weighted by client data size)
        new_weights = {}
        for key in global_weights.keys():
            new_weights[key] = sum([w[key]*sz for w, sz in zip(local_weights, local_sizes)]) / sum(local_sizes)
        global_weights = new_weights
        global_model.load_state_dict(global_weights)

        # Evaluate global model
        global_model.eval()
        with torch.no_grad():
            outputs = global_model(X_test_tensor)
            preds = (outputs > 0.5).float()
            loss_test = binary_cross_entropy(outputs, y_test_tensor)
            acc_test = (preds == y_test_tensor).float().mean().item()
        records.append({'round': round, 'loss': loss_test.item(), 'accuracy': acc_test})
        print(f"Poisoned {i_poisoned} Round {round}: Test Loss={loss_test.item():.4f}, Test Acc={acc_test:.4f}")

    # Save training log
    df = pd.DataFrame(records)
    save_name_path = os.path.join(SAVE_PATH, f'{NAME_SAVE_update_PATH}.csv')
    df.to_csv(save_name_path, index=False)
    print(f"Training log saved to {save_name_path}")

    # Plot loss and accuracy
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(df['round'], df['loss'], marker='o')
    plt.title('Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Round')
    plt.grid(True)
    plt.ylim(0, 1.1)
    # plt.legend()

    plt.subplot(1,2,2)
    plt.plot(df['round'], df['accuracy'], marker='o')
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Round')
    plt.grid(True)
    plt.ylim(0, 1.1)
    # plt.legend()

    plt.tight_layout()
    save_name_path = os.path.join(SAVE_PATH, f'loss_accuracy.jpg')
    plt.savefig(save_name_path)

    save_name_path = os.path.join(SAVE_all_PATH, f'{NAME_SAVE_update_PATH}_latest.jpg')
    plt.savefig(save_name_path)
    plt.close()
