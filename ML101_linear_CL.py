import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt


def training(client_id, model, xb, yb, X_tensor, y_tensor, return_dict):
    # Record initial model values before update in this batch
    w1_epoch = model.linear.weight.data.numpy().copy()
    b1_epoch = model.linear.bias.data.numpy().copy()

    # Forward pass
    outputs = model(xb)
    preds = (outputs > 0.5).float()
    loss = binary_cross_entropy(outputs, yb)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Manual parameter update
    with torch.no_grad():
        model.linear.weight -= LEARNING_RATE * model.linear.weight.grad
        model.linear.bias -= LEARNING_RATE * model.linear.bias.grad

    # Evaluate on the whole dataset for logging
    with torch.no_grad():
        outputs_all = model(X_tensor)
        preds_all = (outputs_all > 0.5).float()
        loss_all = binary_cross_entropy(outputs_all, y_tensor)
        accuracy_all = (preds_all == y_tensor).float().mean().item()

        grad_w1 = model.linear.weight.grad.data.numpy().copy()
        grad_b1 = model.linear.bias.grad.data.numpy().copy()
        w1_new = model.linear.weight.data.numpy().copy()
        b1_new = model.linear.bias.data.numpy().copy()

        return_dict[client_id] = {
            'w1_epoch': w1_epoch,
            'b1_epoch': b1_epoch,
            'loss_all': loss_all,
            'accuracy_all': accuracy_all,
            'grad_w1': grad_w1,
            'grad_b1': grad_b1,
            'w1_new': w1_new,
            'b1_new': b1_new,
        }

# 2. Define simplest PyTorch model
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # 2 inputs, 1 output

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def binary_cross_entropy(pred, target):
    # Clamp predictions to avoid log(0)
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1 - eps)
    return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()

config = json.load(open("config.json"))
LEARNING_RATE = config.get("LEARNING_RATE", 0.5)
EPOCHS = config.get("EPOCHS", 32)
N_PER_CLASS = config.get("N_PER_CLASS", 32)
N_CLASSES = config.get("N_CLASSES", 2)
N_CLIENTS = config.get("N_CLIENTS", 2)
N_POISONED_CLIENTS = config.get("N_POISONED_CLIENTS", 2)
ORIGINAL_M = config.get("ORIGINAL_M", 0.7)
ORIGINAL_B = config.get("ORIGINAL_B", 0.1)
INITIAL_MODEL_W1 = config.get("INITIAL_MODEL_W1", 0.3)
INITIAL_MODEL_W2 = config.get("INITIAL_MODEL_W2", 0.4)
INITIAL_MODEL_B = config.get("INITIAL_MODEL_B", 0.6)

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# List of (save_path, batch_size) configs
configs = [
    ("ML101_linear_CL_batch", N_PER_CLASS * N_CLASSES * N_CLIENTS),
    # ("ML101_linear_CL_minibatch", 8),
    # ("ML101_linear_CL_SGD", 1),
]

for i_poisoned in range(N_POISONED_CLIENTS+1):

    for NAME_SAVE_PATH, BATCH_SIZE in configs:
        BATCH_SIZE = N_PER_CLASS * N_CLASSES * (N_CLIENTS + i_poisoned)
        NAME_SAVE_update_PATH = NAME_SAVE_PATH+f"_poisoned_{i_poisoned}"
        SAVE_PATH = os.path.join("results",NAME_SAVE_update_PATH)
        # Remove existing directory if it exists
        if os.path.exists(SAVE_PATH):
            shutil.rmtree(SAVE_PATH)
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        SAVE_all_PATH = os.path.join("results",'all')
        os.makedirs(SAVE_all_PATH, exist_ok=True)

        # Initialize model
        model = SimpleLinear()
        with torch.no_grad():
            model.linear.weight[:] = torch.tensor([[INITIAL_MODEL_W1, INITIAL_MODEL_W2]])
            model.linear.bias[:] = torch.tensor([INITIAL_MODEL_B])

        # Save initial weights and bias before training
        initial_w = model.linear.weight.data.numpy().copy()
        initial_b = model.linear.bias.data.numpy().copy()

        # 3. Training with recording
        records = []

        # Save X and y for next use
        DATA_PATH = os.path.join("results","data")
        X = np.empty((0, 2))
        y = np.empty((0,))
        for client_id in range(N_CLIENTS):
            X_i = np.load(os.path.join(DATA_PATH, f"ML100_data_X_client{client_id}.npy"))
            y_i = np.load(os.path.join(DATA_PATH, f"ML100_data_y_client{client_id}.npy"))
            X = np.vstack((X, X_i))
            y = np.concatenate((y, y_i))
        for j_poisoned in range(i_poisoned):
            X_i = np.load(os.path.join(DATA_PATH, f"ML100_poison_X_client{j_poisoned}.npy"))
            y_i = np.load(os.path.join(DATA_PATH, f"ML100_poison_y_client{j_poisoned}.npy"))
            X = np.vstack((X, X_i))
            y = np.concatenate((y, y_i))

        print(f"Loaded data shape: X={X.shape}, y={y.shape}")

        # print shape by class
        for i in range(N_CLASSES):
            print(f"Class {i} count: {np.sum(y == i)}")

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        #print y_tensor shape by class
        for i in range(N_CLASSES):
            print(f"Class {i} count in tensor: {torch.sum(y_tensor == i)}")

        for epoch in range(EPOCHS):

            num_batches = int(np.ceil(len(X_tensor) / BATCH_SIZE))

            for batch_idx, start in enumerate(range(0, len(X_tensor), BATCH_SIZE)):
                end = start + BATCH_SIZE
                xb = X_tensor[start:end]
                yb = y_tensor[start:end]

                client_id = 0
                return_dict = {}
                training( client_id, model, xb, yb, X_tensor, y_tensor, return_dict)
                w1_epoch = return_dict[client_id]['w1_epoch']
                b1_epoch = return_dict[client_id]['b1_epoch']
                loss_all = return_dict[client_id]['loss_all']
                accuracy_all = return_dict[client_id]['accuracy_all']
                grad_w1 = return_dict[client_id]['grad_w1']
                grad_b1 = return_dict[client_id]['grad_b1']
                w1_new = return_dict[client_id]['w1_new']
                b1_new = return_dict[client_id]['b1_new']
                
                epoch_batch = epoch + ((batch_idx+1.0) / (num_batches))

                record = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'epoch_batch': epoch_batch,
                    'w1_before': w1_epoch.flatten()[0],
                    'w2_before': w1_epoch.flatten()[1],
                    'b1_before': b1_epoch[0],
                    'loss': loss_all.item(),
                    'accuracy': accuracy_all,
                    'grad_w1': grad_w1.flatten()[0],
                    'grad_w2': grad_w1.flatten()[1],
                    'grad_b1': grad_b1[0],
                    'w1_after': w1_new.flatten()[0],
                    'w2_after': w1_new.flatten()[1],
                    'b1_after': b1_new[0]
                }
                records.append(record)
            
            # Evaluate global model on all data
            with torch.no_grad():
                outputs_all = model(X_tensor)
                preds_all = (outputs_all > 0.5).float()
                loss_all = binary_cross_entropy(outputs_all, y_tensor)
                accuracy_all = (preds_all == y_tensor).float().mean().item()
                w1_epoch_after = model.linear.weight.data.numpy().copy()
                b1_epoch_after = model.linear.bias.data.numpy().copy()

            record = {
                'epoch': epoch+1,
                'batch': -1,
                'epoch_batch': epoch+1,
                'w1_before': w1_epoch.flatten()[0],
                'w2_before': w1_epoch.flatten()[1],
                'b1_before': b1_epoch[0],
                'loss': loss_all.item(),
                'accuracy': accuracy_all,
                'grad_w1': grad_w1.flatten()[0],
                'grad_w2': grad_w1.flatten()[1],
                'grad_b1': grad_b1[0],
                'w1_after': w1_new.flatten()[0],
                'w2_after': w1_new.flatten()[1],
                'b1_after': b1_new[0]
            }
            records.append(record)

        # Plotting by batch (same as before, but inside the loop)
        for i, rec in enumerate(records):
            if rec['batch'] != -1:
                continue
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharex=False)

            # Plot data points by class
            
            for j in range(N_CLASSES):
                ax1.scatter(X[y == j, 0], X[y == j, 1], label=f'Class {j}', alpha=0.5)
            # ax1.scatter(X[:N_PER_CLASS, 0], X[:N_PER_CLASS, 1], color='blue', label='Class 0')
            # ax1.scatter(X[N_PER_CLASS:, 0], X[N_PER_CLASS:, 1], color='red', label='Class 1')
            # Plot decision boundary: w1*x + w2*y + b = 0 => y = -(w1*x + b)/w2
            w1 = rec['w1_before']
            w2 = rec['w2_before']
            b1 = rec['b1_before']
            x_vals = np.array([X[:,0].min()-0.1, X[:,0].max()+0.1])
            if abs(w2) > 1e-6:
                y_vals = -(w1 * x_vals + b1) / w2
                ax1.plot(x_vals, y_vals, 'k--', label='Model boundary')
            # Plot true boundary
            ax1.plot(x_vals, ORIGINAL_M*x_vals + ORIGINAL_B, 'g-', label='True boundary')
            ax1.set_title(f'Epoch {rec["epoch"]} Batch {rec["batch"]} | Loss: {rec["loss"]:.3f} | Acc: {rec["accuracy"]:.2f}')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.legend()
            ax1.set_xlim(X[:,0].min()-0.1, X[:,0].max()+0.1)
            ax1.set_ylim(X[:,1].min()-0.1, X[:,1].max()+0.1)

            # All records up to current
            epochs_range = [r['epoch_batch'] for r in records[:i+1]]
            losses = [r['loss'] for r in records[:i+1]]
            accs = [r['accuracy'] for r in records[:i+1]]

            # Only batch==0 records up to current
            batch0_epochs = [r['epoch_batch'] for r in records[:i+1] if r['batch'] == -1]
            batch0_losses = [r['loss'] for r in records[:i+1] if r['batch'] == -1]
            batch0_accs = [r['accuracy'] for r in records[:i+1] if r['batch'] == -1]

            ax2.plot(epochs_range, losses, marker='o', color='green', label='Batch')
            ax2.plot(batch0_epochs, batch0_losses, marker='x', color='black', linestyle='--', label='Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch.Batch')
            ax2.set_title('Training Loss')
            ax2.grid(True)
            ax2.set_ylim(0, 1.1)
            ax2.legend()

            ax3.plot(epochs_range, accs, marker='o', color='green', label='Batch')
            ax3.plot(batch0_epochs, batch0_accs, marker='x', color='black', linestyle='--', label='Epoch')
            ax3.set_xlabel('Epoch.Batch')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Training Accuracy')
            ax3.grid(True)
            ax3.set_ylim(0, 1.1)
            ax3.legend()

            plt.tight_layout()
            batch_str = f'{rec["batch"]:02d}' if rec["batch"] != -1 else 'last'
            save_name_path = os.path.join(SAVE_PATH, f'epoch_{rec["epoch"]:02d}_batch_{batch_str}.jpg')
            plt.savefig(save_name_path)
            save_name_path = os.path.join(SAVE_all_PATH, f'{NAME_SAVE_update_PATH}_lastest.jpg')


            plt.savefig(save_name_path)
            plt.close()

        # Save to CSV
        df = pd.DataFrame(records)

        save_name_path = os.path.join(SAVE_PATH, f'{NAME_SAVE_update_PATH}.csv')
        df.to_csv(save_name_path, index=False)
        print(f"Training log saved to {save_name_path}")