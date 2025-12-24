"""
Calculate and compare parameter counts for CNN vs LR models
Used to generate accurate documentation for MODEL_TYPE_COMPARISON.md
"""

import torch
from models import ScalableCNN, LogisticRegression

def count_params_by_layer(model):
    """Count parameters for each layer"""
    layer_counts = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_counts.append({
                'name': name,
                'params': param.numel(),
                'shape': list(param.shape)
            })
    return layer_counts

def compare_models(dataset, width_factor, depth):
    """Compare CNN vs LR for given configuration"""
    # Dataset-specific parameters
    if dataset == 'mnist':
        in_channels, img_size = 1, 28
    else:  # cifar10
        in_channels, img_size = 3, 32
    
    # Create models
    cnn = ScalableCNN(num_classes=10, width_factor=width_factor, depth=depth, 
                      in_channels=in_channels, img_size=img_size)
    lr = LogisticRegression(num_classes=10, width_factor=width_factor, depth=depth,
                           in_channels=in_channels, img_size=img_size)
    
    # Count parameters
    cnn_params = cnn.get_num_parameters()
    lr_params = lr.get_num_parameters()
    
    # Layer-wise breakdown
    cnn_layers = count_params_by_layer(cnn)
    lr_layers = count_params_by_layer(lr)
    
    return {
        'dataset': dataset,
        'width_factor': width_factor,
        'depth': depth,
        'cnn_total': cnn_params,
        'lr_total': lr_params,
        'cnn_layers': cnn_layers,
        'lr_layers': lr_layers
    }

if __name__ == '__main__':
    print("=" * 80)
    print("MODEL PARAMETER COMPARISON: CNN vs LR (Logistic Regression/MLP)")
    print("=" * 80)
    
    # Test configurations from experiments
    configs = [
        # MNIST configurations
        ('mnist', 1, 1),
        ('mnist', 1, 4),
        ('mnist', 4, 4),
        ('mnist', 16, 16),
        ('mnist', 64, 64),
        
        # CIFAR-10 configurations
        ('cifar10', 1, 1),
        ('cifar10', 1, 4),
        ('cifar10', 4, 4),
        ('cifar10', 16, 16),
        ('cifar10', 64, 64),
    ]
    
    for dataset, width, depth in configs:
        result = compare_models(dataset, width, depth)
        
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset.upper()}, Width Factor: {width}, Depth: {depth}")
        print(f"{'='*80}")
        print(f"CNN Total Parameters: {result['cnn_total']:,}")
        print(f"LR  Total Parameters: {result['lr_total']:,}")
        print(f"Ratio (LR/CNN): {result['lr_total'] / result['cnn_total']:.2f}x")
        
        print(f"\n--- CNN Layer-by-Layer ---")
        for layer in result['cnn_layers']:
            print(f"  {layer['name']:40s}: {layer['params']:>10,} params, shape={layer['shape']}")
        
        print(f"\n--- LR Layer-by-Layer ---")
        for layer in result['lr_layers']:
            print(f"  {layer['name']:40s}: {layer['params']:>10,} params, shape={layer['shape']}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Parameter Scaling Analysis")
    print("=" * 80)
    
    # Generate summary table
    print("\n### MNIST Parameter Comparison")
    print(f"{'Width':<8}{'Depth':<8}{'CNN Params':<15}{'LR Params':<15}{'Ratio (LR/CNN)':<15}")
    print("-" * 61)
    for dataset, width, depth in configs:
        if dataset == 'mnist':
            result = compare_models(dataset, width, depth)
            ratio = result['lr_total'] / result['cnn_total']
            print(f"{width:<8}{depth:<8}{result['cnn_total']:<15,}{result['lr_total']:<15,}{ratio:<15.2f}x")
    
    print("\n### CIFAR-10 Parameter Comparison")
    print(f"{'Width':<8}{'Depth':<8}{'CNN Params':<15}{'LR Params':<15}{'Ratio (LR/CNN)':<15}")
    print("-" * 61)
    for dataset, width, depth in configs:
        if dataset == 'cifar10':
            result = compare_models(dataset, width, depth)
            ratio = result['lr_total'] / result['cnn_total']
            print(f"{width:<8}{depth:<8}{result['cnn_total']:<15,}{result['lr_total']:<15,}{ratio:<15.2f}x")
