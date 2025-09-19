<<<<<<< HEAD
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from typing import Dict, Any

class OptimizerFactory:
    @staticmethod
    def create_optimizer(model_params, config: Dict[str, Any]):
        """Create optimizer based on config"""
        optimizer_name = config.get('optimizer', 'adamw').lower()
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        return optimizer
    
    @staticmethod
    def create_scheduler(optimizer, config: Dict[str, Any]):
        """Create scheduler based on config"""
        scheduler_name = config.get('scheduler', 'reduce_lr_on_plateau').lower()
        
        if scheduler_name == 'reduce_lr_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=config.get('scheduler_patience', 3),
                factor=config.get('scheduler_factor', 0.5),
                min_lr=config.get('scheduler_min_lr', 1e-6),
                verbose=True
            )
        elif scheduler_name == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=config.get('scheduler_step_size', 10),
                gamma=config.get('scheduler_gamma', 0.5)
            )
        else:
            scheduler = None
            
=======
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from typing import Dict, Any

class OptimizerFactory:
    @staticmethod
    def create_optimizer(model_params, config: Dict[str, Any]):
        """Create optimizer based on config"""
        optimizer_name = config.get('optimizer', 'adamw').lower()
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        return optimizer
    
    @staticmethod
    def create_scheduler(optimizer, config: Dict[str, Any]):
        """Create scheduler based on config"""
        scheduler_name = config.get('scheduler', 'reduce_lr_on_plateau').lower()
        
        if scheduler_name == 'reduce_lr_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=config.get('scheduler_patience', 3),
                factor=config.get('scheduler_factor', 0.5),
                min_lr=config.get('scheduler_min_lr', 1e-6),
                verbose=True
            )
        elif scheduler_name == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=config.get('scheduler_step_size', 10),
                gamma=config.get('scheduler_gamma', 0.5)
            )
        else:
            scheduler = None
            
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
        return scheduler