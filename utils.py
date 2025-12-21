import torch
import torch.nn as nn
import copy
import numpy as np
import logging

class EarlyStopping:
    """Class สำหรับจัดการ Early Stopping"""
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_w = None

    def __call__(self, val_loss, model_weights):
        # Handle NaN values
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.counter += 1
            if self.verbose:
                logging.warning(f"⚠️ NaN/Inf validation loss detected! Counter: {self.counter}/{self.patience}")
                print(f"⚠️ WARNING: NaN/Inf validation loss! EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logging.error("🛑 Early stopping triggered due to persistent NaN/Inf values")
            return
        
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_w = copy.deepcopy(model_weights)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_w = copy.deepcopy(model_weights)
            self.counter = 0
            
    def get_best_weights(self):
        if self.best_model_w is None:
            raise RuntimeError("No best weights saved. Early stopping did not improve model.")
        return self.best_model_w

def train_client(model, train_loader, epochs, lr, device, momentum=0.9, weight_decay=0, max_grad_norm=1.0):
    """ฟังก์ชันเทรนสำหรับ Client 1 ราย
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        epochs: Number of local epochs
        lr: Learning rate
        device: Training device (cpu/cuda)
        momentum: SGD momentum
        weight_decay: L2 regularization
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0, set to None to disable)
    
    Returns:
        model.state_dict(): Trained model weights
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    nan_detected = False
    extreme_gradient_counter = 0
    extreme_gradient_threshold = 10  # Stop after 10 extreme gradients
    
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            try:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
            except (ValueError, TypeError) as e:
                logging.warning(f"Error unpacking batch: {e}")
                continue
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            
            # NaN Detection in Loss
            if torch.isnan(loss) or torch.isinf(loss):
                nan_detected = True
                logging.error(f"🚨 NaN/Inf detected during training! Epoch {epoch+1}, Batch {batch_idx}")
                logging.error(f"   Loss value: {loss.item()}")
                logging.error(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                logging.error(f"   Input range: [{images.min().item():.3f}, {images.max().item():.3f}]")
                # Skip this batch
                continue
            
            loss.backward()
            
            # Gradient Clipping with silent extreme gradient tracking
            if max_grad_norm is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Silently count extreme gradients (no warnings to avoid spam)
                if total_norm > max_grad_norm * 10:
                    extreme_gradient_counter += 1
                    
                    # Abort if too many extreme gradients (no intermediate logging)
                    if extreme_gradient_counter >= extreme_gradient_threshold:
                        logging.error(f"🛑 Training aborted: {extreme_gradient_threshold} extreme gradients detected.")
                        nan_detected = True
                        break
                
                # Check for NaN in gradients
                for name, param in model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        logging.error(f"🚨 NaN/Inf in gradients! Layer: {name}")
                        nan_detected = True
                        break
                
                if nan_detected:
                    break
            
            optimizer.step()
            
            # Check for NaN/Inf in model weights after update
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logging.error(f"🚨 NaN/Inf in weights! Layer: {name}")
                    nan_detected = True
                    break
            
            if nan_detected:
                break
        
        if nan_detected:
            logging.error("🛑 Training stopped: NaN/Inf or severe instability detected.")
            break
            
    return model.state_dict()

def evaluate_model(model, data_loader, device):
    """ฟังก์ชันวัดผล (ใช้ได้ทั้ง Validation และ Test)
    
    Returns:
        avg_loss: Average loss (returns np.nan if NaN detected)
        accuracy: Accuracy (returns 0.0 if NaN detected)
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    nan_detected = False
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
            except ValueError:
                logging.warning(f"Error: DataLoader format mismatch.")
                continue
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                nan_detected = True
                logging.error(f"🚨 NaN/Inf detected in model outputs during evaluation! Batch {batch_idx}")
                logging.error(f"   Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                logging.error(f"   Input range: [{images.min().item():.3f}, {images.max().item():.3f}]")
                # Check model weights
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        logging.error(f"   Layer '{name}' contains NaN/Inf values!")
                        break
                return np.nan, 0.0
            
            loss = criterion(outputs, labels)
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                nan_detected = True
                logging.error(f"🚨 NaN/Inf detected in loss during evaluation! Batch {batch_idx}")
                logging.error(f"   Loss value: {loss.item()}")
                return np.nan, 0.0
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total == 0:
        logging.warning("No samples processed during evaluation")
        return np.nan, 0.0
            
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Final NaN check
    if np.isnan(avg_loss) or np.isinf(avg_loss):
        logging.error(f"🚨 Final avg_loss is NaN/Inf: {avg_loss}")
        return np.nan, accuracy
    
    return avg_loss, accuracy

def fed_avg(weights_list):
    """Aggregate แบบ Federated Averaging (Average Weights)"""
    w_avg = copy.deepcopy(weights_list[0])
    for key in w_avg.keys():
        for i in range(1, len(weights_list)):
            w_avg[key] += weights_list[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weights_list))
    return w_avg

def fed_median(weights_list):
    """
    Robust Aggregation: Coordinate-wise Median
    ป้องกัน Poisoning โดยเลือกค่ามัธยฐานของแต่ละ weight parameter แทนค่าเฉลี่ย
    """
    w_median = copy.deepcopy(weights_list[0])
    
    # Get device from first weight
    device = weights_list[0][list(weights_list[0].keys())[0]].device
    
    # Loop ผ่านทุก layer key (conv1.weight, fc.bias, etc.)
    for key in w_median.keys():
        # Stack weights จากทุก client: shape (num_clients, param_shape...)
        stacked_weights = torch.stack([w[key] for w in weights_list], dim=0)
        
        # คำนวณ Median ตามแกน 0 (แกน Client)
        median_val, _ = torch.median(stacked_weights, dim=0)
        w_median[key] = median_val
        
    return w_median