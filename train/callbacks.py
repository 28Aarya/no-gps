<<<<<<< HEAD
import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CheckpointCallback:
    """Save and load model checkpoints"""
    
    def __init__(self, save_dir: str, save_best: bool = True, save_last: bool = True):
        self.save_dir = Path(save_dir)
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # If it's a file, remove it and create directory
            if self.save_dir.is_file():
                self.save_dir.unlink()
                self.save_dir.mkdir(parents=True, exist_ok=True)
            else:
                # It's already a directory, continue
                pass
        self.save_best = save_best
        self.save_last = save_last
        self.best_metric = float('inf')
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, 
                    metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric
        }
        
        # Save last checkpoint
        if self.save_last:
            torch.save(checkpoint, self.save_dir / 'last_checkpoint.pth')
            
        # Save best checkpoint
        if self.save_best and is_best:
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pth')
            self.best_metric = metrics.get('val_loss', float('inf'))
            logger.info(f"Saved best checkpoint with val_loss: {self.best_metric:.6f}")
    
    def load_checkpoint(self, model, optimizer, scheduler, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint if it exists"""
        last_checkpoint = self.save_dir / 'last_checkpoint.pth'
        if last_checkpoint.exists():
            return str(last_checkpoint)
        return None

class EarlyStoppingCallback:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        """Return True if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.patience} epochs")
            return True
        return False

class LoggingCallback:
    """Logging callback for training progress"""
    
    def __init__(self, log_interval: int = 10, log_file: str = None):
        self.log_interval = log_interval
        self.log_file = log_file
        
        # Set up file logging if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                val_metrics: Dict[str, float], lr: float):
        """Log epoch metrics"""
        if epoch % self.log_interval == 0:
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics.get('loss', 0):.6f} | "
                f"Val Loss: {val_metrics.get('val_loss', 0):.6f} | "
                f"LR: {lr:.2e}"
            )
=======
import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CheckpointCallback:
    """Save and load model checkpoints"""
    
    def __init__(self, save_dir: str, save_best: bool = True, save_last: bool = True):
        self.save_dir = Path(save_dir)
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # If it's a file, remove it and create directory
            if self.save_dir.is_file():
                self.save_dir.unlink()
                self.save_dir.mkdir(parents=True, exist_ok=True)
            else:
                # It's already a directory, continue
                pass
        self.save_best = save_best
        self.save_last = save_last
        self.best_metric = float('inf')
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, 
                    metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric
        }
        
        # Save last checkpoint
        if self.save_last:
            torch.save(checkpoint, self.save_dir / 'last_checkpoint.pth')
            
        # Save best checkpoint
        if self.save_best and is_best:
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pth')
            self.best_metric = metrics.get('val_loss', float('inf'))
            logger.info(f"Saved best checkpoint with val_loss: {self.best_metric:.6f}")
    
    def load_checkpoint(self, model, optimizer, scheduler, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint if it exists"""
        last_checkpoint = self.save_dir / 'last_checkpoint.pth'
        if last_checkpoint.exists():
            return str(last_checkpoint)
        return None

class EarlyStoppingCallback:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        """Return True if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.patience} epochs")
            return True
        return False

class LoggingCallback:
    """Logging callback for training progress"""
    
    def __init__(self, log_interval: int = 10, log_file: str = None):
        self.log_interval = log_interval
        self.log_file = log_file
        
        # Set up file logging if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                val_metrics: Dict[str, float], lr: float):
        """Log epoch metrics"""
        if epoch % self.log_interval == 0:
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics.get('loss', 0):.6f} | "
                f"Val Loss: {val_metrics.get('val_loss', 0):.6f} | "
                f"LR: {lr:.2e}"
            )
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
