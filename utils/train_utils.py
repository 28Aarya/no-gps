<<<<<<< HEAD
# train/utils.py
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from utils.shapes import expect_shape, assert_finite

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_batch(batch: Tuple[torch.Tensor, torch.Tensor], model, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate and prepare batch for training"""
    x, y = batch
    if x.device != torch.device(device):
        x = x.to(device)
    if y.device != torch.device(device):
        y = y.to(device)
    
    # Shape validation for PatchTST
    if x.dim() != 4:
        raise ValueError(f"Expected x shape (B, C, P, patch_len). Got: {tuple(x.shape)}")
    if y.dim() != 3:
        raise ValueError(f"Expected y shape (B, pred_len, O). Got: {tuple(y.shape)}")
    
    # Data validation
    assert_finite(x, "batch_input")
    assert_finite(y, "batch_labels")
    
    return x, y

def calculate_gradient_norm(model) -> float:
    """Calculate gradient norm for monitoring"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def check_cuda_availability():
    """Check CUDA availability and print debug info"""
    print("=== CUDA Debug Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA is not available!")
        print("Possible reasons:")
        print("1. PyTorch not installed with CUDA support")
        print("2. No NVIDIA GPU detected")
        print("3. CUDA drivers not installed")
    
    print("\n=== Test Tensor Creation ===")
    try:
        x = torch.randn(2, 3)
        print(f"CPU tensor: {x.device}")
        
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            print(f"CUDA tensor: {x_cuda.device}")
        else:
            print("Cannot create CUDA tensor - CUDA not available")
    except Exception as e:
        print(f"Error creating tensors: {e}")
=======
# train/utils.py
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from utils.shapes import expect_shape, assert_finite

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_batch(batch: Tuple[torch.Tensor, torch.Tensor], model, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate and prepare batch for training"""
    x, y = batch
    if x.device != torch.device(device):
        x = x.to(device)
    if y.device != torch.device(device):
        y = y.to(device)
    
    # Shape validation for PatchTST
    if x.dim() != 4:
        raise ValueError(f"Expected x shape (B, C, P, patch_len). Got: {tuple(x.shape)}")
    if y.dim() != 3:
        raise ValueError(f"Expected y shape (B, pred_len, O). Got: {tuple(y.shape)}")
    
    # Data validation
    assert_finite(x, "batch_input")
    assert_finite(y, "batch_labels")
    
    return x, y

def calculate_gradient_norm(model) -> float:
    """Calculate gradient norm for monitoring"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def check_cuda_availability():
    """Check CUDA availability and print debug info"""
    print("=== CUDA Debug Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA is not available!")
        print("Possible reasons:")
        print("1. PyTorch not installed with CUDA support")
        print("2. No NVIDIA GPU detected")
        print("3. CUDA drivers not installed")
    
    print("\n=== Test Tensor Creation ===")
    try:
        x = torch.randn(2, 3)
        print(f"CPU tensor: {x.device}")
        
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            print(f"CUDA tensor: {x_cuda.device}")
        else:
            print("Cannot create CUDA tensor - CUDA not available")
    except Exception as e:
        print(f"Error creating tensors: {e}")
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
