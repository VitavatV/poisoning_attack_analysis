"""
Compare CNN and LR model parameters layer by layer
"""
import torch
from models import ScalableCNN, LogisticRegression

def count_parameters(model, model_name):
    """Print detailed parameter count for each layer"""
    print(f"\n{'='*70}")
    print(f"{model_name} - Layer-by-Layer Parameter Count")
    print(f"{'='*70}")
    
    total_params = 0
    
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            
            # Format parameter shape
            shape_str = str(list(param.shape))
            
            print(f"Layer {i+1:2d}: {name:40s} | Shape: {shape_str:20s} | Params: {num_params:,}")
    
    print(f"{'-'*70}")
    print(f"{'TOTAL PARAMETERS':62s} | {total_params:,}")
    print(f"{'='*70}\n")
    
    return total_params

def detailed_layer_calculation(model, model_name, config):
    """Show detailed calculation for each layer"""
    print(f"\n{'='*70}")
    print(f"{model_name} - Detailed Parameter Calculations")
    print(f"Configuration: {config}")
    print(f"{'='*70}\n")
    
    if isinstance(model, ScalableCNN):
        print("CNN Layer Calculations:")
        print("-" * 70)
        
        in_ch = config['in_channels']
        wf = config['width_factor']
        depth = config['depth']
        
        for i in range(depth):
            out_ch = wf
            
            # Conv2d parameters: (in_channels × out_channels × kernel_h × kernel_w) + bias
            conv_params = (in_ch * out_ch * 3 * 3) + out_ch
            bn_params = out_ch * 2  # BatchNorm: gamma and beta
            
            print(f"\nBlock {i+1}:")
            print(f"  Conv2d({in_ch} → {out_ch}, 3×3):")
            print(f"    Weights: {in_ch} × {out_ch} × 3 × 3 = {in_ch * out_ch * 3 * 3:,}")
            print(f"    Bias:    {out_ch}")
            print(f"    Total:   {conv_params:,}")
            print(f"  BatchNorm2d({out_ch}):")
            print(f"    Params:  {bn_params} (γ, β)")
            print(f"  ReLU: 0 parameters")
            
            if i % 2 != 0:
                print(f"  MaxPool2d: 0 parameters")
            
            in_ch = out_ch
        
        # Calculate final FC layer
        # Need to determine spatial dimension after MaxPools
        img_size = config['img_size']
        num_pools = depth // 2
        final_spatial = img_size // (2 ** num_pools)
        flatten_dim = wf * (final_spatial ** 2)
        fc_params = (flatten_dim * config['num_classes']) + config['num_classes']
        
        print(f"\nFinal Classifier:")
        print(f"  Flatten dimension: {wf} × {final_spatial} × {final_spatial} = {flatten_dim:,}")
        print(f"  Linear({flatten_dim} → {config['num_classes']}):")
        print(f"    Weights: {flatten_dim} × {config['num_classes']} = {flatten_dim * config['num_classes']:,}")
        print(f"    Bias:    {config['num_classes']}")
        print(f"    Total:   {fc_params:,}")
    
    elif isinstance(model, LogisticRegression):
        print("LR (MLP) Layer Calculations:")
        print("-" * 70)
        
        input_dim = config['in_channels'] * config['img_size'] * config['img_size']
        wf = config['width_factor']
        depth = config['depth']
        
        print(f"\nInput dimension: {config['in_channels']} × {config['img_size']} × {config['img_size']} = {input_dim:,}")
        
        if depth == 1:
            # Pure logistic regression
            fc_params = (input_dim * config['num_classes']) + config['num_classes']
            print(f"\nLinear({input_dim} → {config['num_classes']}):")
            print(f"  Weights: {input_dim} × {config['num_classes']} = {input_dim * config['num_classes']:,}")
            print(f"  Bias:    {config['num_classes']}")
            print(f"  Total:   {fc_params:,}")
        else:
            # MLP
            current_dim = input_dim
            
            for i in range(depth - 1):
                hidden_size = wf
                fc_params = (current_dim * hidden_size) + hidden_size
                bn_params = hidden_size * 2
                
                print(f"\nHidden Layer {i+1}:")
                print(f"  Linear({current_dim} → {hidden_size}):")
                print(f"    Weights: {current_dim} × {hidden_size} = {current_dim * hidden_size:,}")
                print(f"    Bias:    {hidden_size}")
                print(f"    Total:   {fc_params:,}")
                print(f"  ReLU: 0 parameters")
                print(f"  BatchNorm1d({hidden_size}):")
                print(f"    Params:  {bn_params} (γ, β)")
                
                current_dim = hidden_size
            
            # Output layer
            fc_params = (current_dim * config['num_classes']) + config['num_classes']
            print(f"\nOutput Layer:")
            print(f"  Linear({current_dim} → {config['num_classes']}):")
            print(f"    Weights: {current_dim} × {config['num_classes']} = {current_dim * config['num_classes']:,}")
            print(f"    Bias:    {config['num_classes']}")
            print(f"    Total:   {fc_params:,}")

def main():
    # Test configurations
    configs = [
        {
            'name': 'MNIST',
            'num_classes': 10,
            'width_factor': 4,
            'depth': 4,
            'in_channels': 1,
            'img_size': 28
        },
        {
            'name': 'CIFAR-10',
            'num_classes': 10,
            'width_factor': 4,
            'depth': 4,
            'in_channels': 3,
            'img_size': 32
        }
    ]
    
    for config in configs:
        print("\n" + "="*70)
        print(f"COMPARISON FOR {config['name']}")
        print("="*70)
        
        # Create models
        cnn = ScalableCNN(
            num_classes=config['num_classes'],
            width_factor=config['width_factor'],
            depth=config['depth'],
            in_channels=config['in_channels'],
            img_size=config['img_size']
        )
        
        lr = LogisticRegression(
            num_classes=config['num_classes'],
            width_factor=config['width_factor'],
            depth=config['depth'],
            in_channels=config['in_channels'],
            img_size=config['img_size']
        )
        
        # Count parameters
        cnn_params = count_parameters(cnn, f"ScalableCNN ({config['name']})")
        lr_params = count_parameters(lr, f"LogisticRegression ({config['name']})")
        
        # Detailed calculations
        detailed_layer_calculation(cnn, f"ScalableCNN ({config['name']})", config)
        detailed_layer_calculation(lr, f"LogisticRegression ({config['name']})", config)
        
        # Summary comparison
        print("\n" + "="*70)
        print(f"SUMMARY - {config['name']}")
        print("="*70)
        print(f"CNN Total Parameters:  {cnn_params:,}")
        print(f"LR Total Parameters:   {lr_params:,}")
        print(f"Ratio (LR/CNN):        {lr_params/cnn_params:.2f}x")
        if lr_params > cnn_params:
            print(f"LR has {lr_params - cnn_params:,} MORE parameters than CNN")
        else:
            print(f"CNN has {cnn_params - lr_params:,} MORE parameters than LR")
        print("="*70)

if __name__ == "__main__":
    main()
