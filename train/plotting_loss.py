<<<<<<< HEAD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class TrainingPlotter:
    """Dedicated plotting class for training progress"""
    
    def __init__(self, save_dir: str, plot_interval: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_interval = plot_interval
        
        # Store metrics for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.epochs = []
        
    def update(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Update stored metrics"""
        self.epochs.append(epoch)
        self.train_losses.append(train_metrics.get('loss', 0))
        self.val_losses.append(val_metrics.get('val_loss', 0))
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 3, 1)
        plt.plot(self.epochs, self.train_losses, label='Train Loss', color='blue', linewidth=2)
        plt.plot(self.epochs, self.val_losses, label='Val Loss', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Position error plot
        plt.subplot(2, 3, 2)
        train_pos_errors = [m.get('position_error', 0) for m in self.train_metrics]
        val_pos_errors = [m.get('position_error', 0) for m in self.val_metrics]
        plt.plot(self.epochs, train_pos_errors, label='Train Position Error', color='blue', linewidth=2)
        plt.plot(self.epochs, val_pos_errors, label='Val Position Error', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Position Error')
        plt.title('Position Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Altitude error plot
        plt.subplot(2, 3, 3)
        train_alt_errors = [m.get('altitude_error', 0) for m in self.train_metrics]
        val_alt_errors = [m.get('altitude_error', 0) for m in self.val_metrics]
        plt.plot(self.epochs, train_alt_errors, label='Train Altitude Error', color='blue', linewidth=2)
        plt.plot(self.epochs, val_alt_errors, label='Val Altitude Error', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Altitude Error')
        plt.title('Altitude Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # RMSE plot
        plt.subplot(2, 3, 4)
        train_rmse = [m.get('rmse', 0) for m in self.train_metrics]
        val_rmse = [m.get('rmse', 0) for m in self.val_metrics]
        plt.plot(self.epochs, train_rmse, label='Train RMSE', color='blue', linewidth=2)
        plt.plot(self.epochs, val_rmse, label='Val RMSE', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Root Mean Square Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MAE plot
        plt.subplot(2, 3, 5)
        train_mae = [m.get('mae', 0) for m in self.train_metrics]
        val_mae = [m.get('mae', 0) for m in self.val_metrics]
        plt.plot(self.epochs, train_mae, label='Train MAE', color='blue', linewidth=2)
        plt.plot(self.epochs, val_mae, label='Val MAE', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(2, 3, 6)
        if hasattr(self, 'learning_rates'):
            plt.plot(self.epochs, self.learning_rates, label='Learning Rate', color='green', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {self.save_dir / 'training_plots.png'}")
        
    def save_metrics(self):
        """Save metrics to CSV for later analysis"""
        metrics_df = pd.DataFrame({
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_position_error': [m.get('position_error', 0) for m in self.train_metrics],
            'val_position_error': [m.get('position_error', 0) for m in self.val_metrics],
            'train_altitude_error': [m.get('altitude_error', 0) for m in self.train_metrics],
            'val_altitude_error': [m.get('altitude_error', 0) for m in self.val_metrics],
            'train_rmse': [m.get('rmse', 0) for m in self.train_metrics],
            'val_rmse': [m.get('rmse', 0) for m in self.val_metrics],
            'train_mae': [m.get('mae', 0) for m in self.train_metrics],
            'val_mae': [m.get('mae', 0) for m in self.val_metrics],
        })
        
        metrics_df.to_csv(self.save_dir / 'training_metrics.csv', index=False)
        logger.info(f"Training metrics saved to {self.save_dir / 'training_metrics.csv'}")
    
    def update_learning_rate(self, lr: float):
        """Update learning rate for plotting"""
        if not hasattr(self, 'learning_rates'):
            self.learning_rates = []
        self.learning_rates.append(lr)
=======
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class TrainingPlotter:
    """Dedicated plotting class for training progress"""
    
    def __init__(self, save_dir: str, plot_interval: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_interval = plot_interval
        
        # Store metrics for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.epochs = []
        
    def update(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Update stored metrics"""
        self.epochs.append(epoch)
        self.train_losses.append(train_metrics.get('loss', 0))
        self.val_losses.append(val_metrics.get('val_loss', 0))
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 3, 1)
        plt.plot(self.epochs, self.train_losses, label='Train Loss', color='blue', linewidth=2)
        plt.plot(self.epochs, self.val_losses, label='Val Loss', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Position error plot
        plt.subplot(2, 3, 2)
        train_pos_errors = [m.get('position_error', 0) for m in self.train_metrics]
        val_pos_errors = [m.get('position_error', 0) for m in self.val_metrics]
        plt.plot(self.epochs, train_pos_errors, label='Train Position Error', color='blue', linewidth=2)
        plt.plot(self.epochs, val_pos_errors, label='Val Position Error', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Position Error')
        plt.title('Position Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Altitude error plot
        plt.subplot(2, 3, 3)
        train_alt_errors = [m.get('altitude_error', 0) for m in self.train_metrics]
        val_alt_errors = [m.get('altitude_error', 0) for m in self.val_metrics]
        plt.plot(self.epochs, train_alt_errors, label='Train Altitude Error', color='blue', linewidth=2)
        plt.plot(self.epochs, val_alt_errors, label='Val Altitude Error', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Altitude Error')
        plt.title('Altitude Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # RMSE plot
        plt.subplot(2, 3, 4)
        train_rmse = [m.get('rmse', 0) for m in self.train_metrics]
        val_rmse = [m.get('rmse', 0) for m in self.val_metrics]
        plt.plot(self.epochs, train_rmse, label='Train RMSE', color='blue', linewidth=2)
        plt.plot(self.epochs, val_rmse, label='Val RMSE', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Root Mean Square Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MAE plot
        plt.subplot(2, 3, 5)
        train_mae = [m.get('mae', 0) for m in self.train_metrics]
        val_mae = [m.get('mae', 0) for m in self.val_metrics]
        plt.plot(self.epochs, train_mae, label='Train MAE', color='blue', linewidth=2)
        plt.plot(self.epochs, val_mae, label='Val MAE', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(2, 3, 6)
        if hasattr(self, 'learning_rates'):
            plt.plot(self.epochs, self.learning_rates, label='Learning Rate', color='green', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {self.save_dir / 'training_plots.png'}")
        
    def save_metrics(self):
        """Save metrics to CSV for later analysis"""
        metrics_df = pd.DataFrame({
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_position_error': [m.get('position_error', 0) for m in self.train_metrics],
            'val_position_error': [m.get('position_error', 0) for m in self.val_metrics],
            'train_altitude_error': [m.get('altitude_error', 0) for m in self.train_metrics],
            'val_altitude_error': [m.get('altitude_error', 0) for m in self.val_metrics],
            'train_rmse': [m.get('rmse', 0) for m in self.train_metrics],
            'val_rmse': [m.get('rmse', 0) for m in self.val_metrics],
            'train_mae': [m.get('mae', 0) for m in self.train_metrics],
            'val_mae': [m.get('mae', 0) for m in self.val_metrics],
        })
        
        metrics_df.to_csv(self.save_dir / 'training_metrics.csv', index=False)
        logger.info(f"Training metrics saved to {self.save_dir / 'training_metrics.csv'}")
    
    def update_learning_rate(self, lr: float):
        """Update learning rate for plotting"""
        if not hasattr(self, 'learning_rates'):
            self.learning_rates = []
        self.learning_rates.append(lr)
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
