import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import threading
import errno

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt

def client_thread_fn(client_id, global_model, client_data, epoch, local_logs, local_weights):
    # Each client gets a fresh copy of the global model
    local_model = SimpleLinear()
    set_model_weights(local_model, get_model_weights(global_model))

    X_local, y_local = client_data[client_id]
    num_batches = int(np.ceil(len(X_local) / BATCH_SIZE))

    for batch_idx, start in enumerate(range(0, len(X_local), BATCH_SIZE)):
        end = start + BATCH_SIZE
        xb = X_local[start:end]
        yb = y_local[start:end]

        return_dict = {}
        training(client_id, local_model, xb, yb, X_tensor, y_tensor, return_dict)

        w1_epoch = return_dict[client_id]['w1_epoch']
        b1_epoch = return_dict[client_id]['b1_epoch']
        loss_all = return_dict[client_id]['loss_all']
        accuracy_all = return_dict[client_id]['accuracy_all']
        grad_w1 = return_dict[client_id]['grad_w1']
        grad_b1 = return_dict[client_id]['grad_b1']
        w1_new = return_dict[client_id]['w1_new']
        b1_new = return_dict[client_id]['b1_new']

        epoch_batch = epoch + ((batch_idx + 1.0) / num_batches)

        record = {
            'epoch': epoch,
            'batch': batch_idx,
            'epoch_batch': epoch_batch,
            'client_id': client_id,
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

        local_logs.append(record)

    # Save local model weights after local training
    local_weights.append(get_model_weights(local_model))

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

def get_model_weights(model):
    return [param.data.clone() for param in model.parameters()]

def set_model_weights(model, weights):
    for param, w in zip(model.parameters(), weights):
        param.data.copy_(w)

def average_weights(weights_list):
    avg_weights = []
    for weights in zip(*weights_list):
        avg = torch.stack(weights, dim=0).mean(dim=0)
        avg_weights.append(avg)
    return avg_weights

def get_model_grads(local_model, global_model):
    # Return the difference between local and global model parameters (delta)
    return [
        global_param.data.clone() - local_param.data.clone()
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters())
    ]

def binary_cross_entropy(pred, target):
    # Clamp predictions to avoid log(0)
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1 - eps)
    return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()

def handle_remove_readonly(func, path, exc):
    import stat
    excvalue = exc[1]
    if func in (os.unlink, os.remove, os.rmdir):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass
    elif excvalue.errno == errno.ENOENT:
        pass  # File already deleted
    else:
        print(f"Warning: Could not delete {path}: {excvalue}")



config = json.load(open("config.json"))
LEARNING_RATE = config.get("LEARNING_RATE", 0.5)
EPOCHS = config.get("EPOCHS", 32)
N_PER_CLASS = config.get("N_PER_CLASS", 32)
N_CLASSES = config.get("N_CLASSES", 2)
N_CLIENTS = config.get("N_CLIENTS", 2)
N_TEST = config.get("N_TEST", 8)
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
    ("ML302_linear_FLW_batch", N_PER_CLASS * N_CLASSES),
    # ("ML302_linear_FLW_minibatch", 8),
    # ("ML302_linear_FLW_SGD", 1),
]

# for mode in ['normal', 'uniform', 'laplace', 'beta', 'triangular']:
for mode in ['triangular']:

    for i_poisoned in range(60,N_POISONED_CLIENTS+1):

        for NAME_SAVE_PATH, BATCH_SIZE in configs:
            NAME_SAVE_update_PATH = f"poisoned_{i_poisoned}"
            SAVE_PATH = os.path.join("results",NAME_SAVE_PATH,mode,NAME_SAVE_update_PATH)
            # Remove existing directory if it exists
            if os.path.exists(SAVE_PATH):
                shutil.rmtree(SAVE_PATH, onerror=handle_remove_readonly)
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            SAVE_all_PATH = os.path.join("results",NAME_SAVE_PATH,mode,'all')
            os.makedirs(SAVE_all_PATH, exist_ok=True)

            # Initialize global model
            global_model = SimpleLinear()
            with torch.no_grad():
                global_model.linear.weight[:] = torch.tensor([[0.5, 0.9]])
                global_model.linear.bias[:] = torch.tensor([1.0])

            # 3. Training with recording
            records = []

            # Split data among clients
            DATA_PATH = os.path.join("results","data",mode)
            client_data = [] # This will store a list of (X, y) tuples for each client
            X = np.empty((0, 2))
            y = np.empty((0,))
            for i_client in range(N_CLIENTS): # 4. Loop through each client to assign their data
                
                X_i = np.load(os.path.join(DATA_PATH, f"ML100_data_X_client{i_client}.npy"))
                y_i = np.load(os.path.join(DATA_PATH, f"ML100_data_y_client{i_client}.npy"))
                X = np.vstack((X, X_i))
                y = np.concatenate((y, y_i))
                
                X_tensor = torch.tensor(X_i, dtype=torch.float32)
                y_tensor = torch.tensor(y_i, dtype=torch.float32).unsqueeze(1)

                client_data.append((X_tensor, y_tensor)) # Appends the (X, y) tuple for the current client to the list.

            for j_poisoned in range(i_poisoned):
                
                X_i = np.load(os.path.join(DATA_PATH, f"ML100_poison_X_client{j_poisoned}.npy"))
                y_i = np.load(os.path.join(DATA_PATH, f"ML100_poison_y_client{j_poisoned}.npy"))
                X = np.vstack((X, X_i))
                y = np.concatenate((y, y_i))
                
                X_tensor = torch.tensor(X_i, dtype=torch.float32)
                y_tensor = torch.tensor(y_i, dtype=torch.float32).unsqueeze(1)

                client_data.append((X_tensor, y_tensor)) # Appends the (X, y) tuple for the current client to the list.

            X_test = np.load(os.path.join(DATA_PATH, f"ML100_test_X_client.npy"))
            y_test = np.load(os.path.join(DATA_PATH, f"ML100_test_y_client.npy"))
            
            print(f"Loaded data shape: X={X.shape}, y={y.shape}")

            # print shape by class
            for i in range(N_CLASSES):
                print(f"Class {i} count: {np.sum(y == i)}")

            # X_tensor = torch.tensor(X, dtype=torch.float32)
            # y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

            for epoch in range(EPOCHS):
                local_weights = []
                local_logs = []

                # Create and start threads
                threads = []
                for client_id in range(N_CLIENTS+i_poisoned):
                    t = threading.Thread(target=client_thread_fn, args=(client_id, global_model, client_data, epoch, local_logs, local_weights))
                    threads.append(t)
                    t.start()

                # Wait for all threads to finish
                for t in threads:
                    t.join()

                # for client_id in range(N_CLIENTS):
                #     # Each client gets a fresh copy of the global model
                #     local_model = SimpleLinear()
                #     set_model_weights(local_model, get_model_weights(global_model))

                #     X_local, y_local = client_data[client_id]
                #     num_batches = int(np.ceil(len(X_local) / BATCH_SIZE))

                #     for batch_idx, start in enumerate(range(0, len(X_local), BATCH_SIZE)):
                #         end = start + BATCH_SIZE
                #         xb = X_local[start:end]
                #         yb = y_local[start:end]

                #         return_dict = {}
                #         training(client_id, local_model, xb, yb, X_tensor, y_tensor, return_dict)
                #         w1_epoch = return_dict[client_id]['w1_epoch']
                #         b1_epoch = return_dict[client_id]['b1_epoch']
                #         loss_all = return_dict[client_id]['loss_all']
                #         accuracy_all = return_dict[client_id]['accuracy_all']
                #         grad_w1 = return_dict[client_id]['grad_w1']
                #         grad_b1 = return_dict[client_id]['grad_b1']
                #         w1_new = return_dict[client_id]['w1_new']
                #         b1_new = return_dict[client_id]['b1_new']

                #         epoch_batch = epoch + ((batch_idx+1.0) / (num_batches))

                #         record = {
                #             'epoch': epoch,
                #             'batch': batch_idx,
                #             'epoch_batch': epoch_batch,
                #             'client_id': client_id,
                #             'w1_before': w1_epoch.flatten()[0],
                #             'w2_before': w1_epoch.flatten()[1],
                #             'b1_before': b1_epoch[0],
                #             'loss': loss_all.item(),
                #             'accuracy': accuracy_all,
                #             'grad_w1': grad_w1.flatten()[0],
                #             'grad_w2': grad_w1.flatten()[1],
                #             'grad_b1': grad_b1[0],
                #             'w1_after': w1_new.flatten()[0],
                #             'w2_after': w1_new.flatten()[1],
                #             'b1_after': b1_new[0]
                #         }
                #         local_logs.append(record)

                #     # Save local model weights after local training
                #     local_weights.append(get_model_weights(local_model))

                w1_epoch_before = global_model.linear.weight.data.numpy().copy()
                b1_epoch_before = global_model.linear.bias.data.numpy().copy()

                ori_global_model = SimpleLinear()
                set_model_weights(ori_global_model, get_model_weights(global_model))
                # Federated weight averaging
                avg_weights = average_weights(local_weights)
                set_model_weights(global_model, avg_weights)
                grads = get_model_grads(global_model, ori_global_model)
                for param, grad in zip(ori_global_model.parameters(), grads):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param.data)
                    param.grad.copy_(grad)
                grad_w1 = ori_global_model.linear.weight.grad.data.numpy().copy()
                grad_b1 = ori_global_model.linear.bias.grad.data.numpy().copy()


                # Evaluate global model on all data
                with torch.no_grad():
                    outputs_all = global_model(X_test_tensor)
                    preds_all = (outputs_all > 0.5).float()
                    loss_all = binary_cross_entropy(outputs_all, y_test_tensor)
                    accuracy_all = (preds_all == y_test_tensor).float().mean().item()
                    w1_epoch_after = global_model.linear.weight.data.numpy().copy()
                    b1_epoch_after = global_model.linear.bias.data.numpy().copy()

                record = {
                    'epoch': epoch+1,
                    'batch': -1,
                    'epoch_batch': epoch+1,
                    'client_id': -1,
                    'w1_before': w1_epoch_before.flatten()[0],
                    'w2_before': w1_epoch_before.flatten()[1],
                    'b1_before': b1_epoch_before[0],
                    'loss': loss_all.item(),
                    'accuracy': accuracy_all,
                    'grad_w1': grad_w1.flatten()[0],
                    'grad_w2': grad_w1.flatten()[1],
                    'grad_b1': grad_b1[0],
                    'w1_after': w1_epoch_after.flatten()[0],
                    'w2_after': w1_epoch_after.flatten()[1],
                    'b1_after': b1_epoch_after[0]
                }
                records.extend(local_logs)
                records.append(record)

            # Plotting by batch (same as before, but inside the loop)
            for i_rec, rec in enumerate(records):
                if rec['batch'] != -1:
                    continue
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharex=False)

                # Plot data points by class
                
                for j in range(N_CLASSES):
                    ax1.scatter(X_test[y_test == j, 0], X_test[y_test == j, 1], label=f'Class {j}', alpha=0.5)
                # ax1.scatter(X[:N_PER_CLASS, 0], X[:N_PER_CLASS, 1], color='blue', label='Class 0')
                # ax1.scatter(X[N_PER_CLASS:, 0], X[N_PER_CLASS:, 1], color='red', label='Class 1')
                # Plot decision boundary: w1*x + w2*y + b = 0 => y = -(w1*x + b)/w2
                w1 = rec['w1_before']
                w2 = rec['w2_before']
                b1 = rec['b1_before']
                x_vals = np.array([X_test[:,0].min()-0.1, X_test[:,0].max()+0.1])
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
                epochs_range = {}
                losses = {}
                accs = {}
                for i_client in range(N_CLIENTS):
                    epochs_range[i_client] = [r['epoch_batch'] for r in records[:i_rec+1] if r['client_id'] == i_client]
                    losses[i_client] = [r['loss'] for r in records[:i_rec+1] if r['client_id'] == i_client]
                    accs[i_client] = [r['accuracy'] for r in records[:i_rec+1] if r['client_id'] == i_client]

                # Only batch==0 records up to current
                batch0_epochs = [r['epoch_batch'] for r in records[:i_rec+1] if r['batch'] == -1]
                batch0_losses = [r['loss'] for r in records[:i_rec+1] if r['batch'] == -1]
                batch0_accs = [r['accuracy'] for r in records[:i_rec+1] if r['batch'] == -1]

                for i_client in range(N_CLIENTS):
                    ax2.plot(epochs_range[i_client], losses[i_client], marker='o', color=f'C{i_client}', label=f'Batch, client {i_client}')
                ax2.plot(batch0_epochs, batch0_losses, marker='x', color='black', linestyle='--', label='Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_xlabel('Epoch.Batch')
                ax2.set_title('Training Loss')
                ax2.grid(True)
                ax2.set_ylim(0, 1.1)
                # ax2.legend()

                for i_client in range(N_CLIENTS):
                    ax3.plot(epochs_range[i_client], accs[i_client], marker='o', color=f'C{i_client}', label=f'Batch, client {i_client}')
                ax3.plot(batch0_epochs, batch0_accs, marker='x', color='black', linestyle='--', label='Epoch')
                ax3.set_xlabel('Epoch.Batch')
                ax3.set_ylabel('Accuracy')
                ax3.set_title('Training Accuracy')
                ax3.grid(True)
                ax3.set_ylim(0, 1.1)
                # ax3.legend()

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