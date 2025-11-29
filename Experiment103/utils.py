import torch
import torch.nn as nn
import copy
import numpy as np

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

def train_client(model, train_loader, epochs, lr, device, momentum=0.9, weight_decay=0):
    """ฟังก์ชันเทรนสำหรับ Client 1 ราย"""
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        for batch in train_loader:
            try:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
            except (ValueError, TypeError) as e:
                print(f"Error unpacking batch: {e}")
                continue
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
    return model.state_dict()

def evaluate_model(model, data_loader, device):
    """ฟังก์ชันวัดผล (ใช้ได้ทั้ง Validation และ Test)"""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            try:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
            except ValueError:
                print(f"Error: DataLoader format mismatch.")
                continue
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
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