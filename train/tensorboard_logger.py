<<<<<<< HEAD
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class TensorboardLogger:
    """Dedicated TensorBoard logging class"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info(f"TensorBoard logging to {self.log_dir}")
        
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                val_metrics: Dict[str, float], lr: float):
        """Log metrics to TensorBoard"""
        # Losses
        self.writer.add_scalar('Loss/Train', train_metrics.get('loss', 0), epoch)
        self.writer.add_scalar('Loss/Validation', val_metrics.get('val_loss', 0), epoch)
        
        # Learning rate
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Position errors
        self.writer.add_scalar('Position_Error/Train', train_metrics.get('position_error', 0), epoch)
        self.writer.add_scalar('Position_Error/Validation', val_metrics.get('position_error', 0), epoch)
        
        # Altitude errors
        self.writer.add_scalar('Altitude_Error/Train', train_metrics.get('altitude_error', 0), epoch)
        self.writer.add_scalar('Altitude_Error/Validation', val_metrics.get('altitude_error', 0), epoch)
        
        # RMSE
        self.writer.add_scalar('RMSE/Train', train_metrics.get('rmse', 0), epoch)
        self.writer.add_scalar('RMSE/Validation', val_metrics.get('rmse', 0), epoch)
        
        # MAE
        self.writer.add_scalar('MAE/Train', train_metrics.get('mae', 0), epoch)
        self.writer.add_scalar('MAE/Validation', val_metrics.get('mae', 0), epoch)
        
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()
        logger.info("TensorBoard logger closed")
=======
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class TensorboardLogger:
    """Dedicated TensorBoard logging class"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info(f"TensorBoard logging to {self.log_dir}")
        
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                val_metrics: Dict[str, float], lr: float):
        """Log metrics to TensorBoard"""
        # Losses
        self.writer.add_scalar('Loss/Train', train_metrics.get('loss', 0), epoch)
        self.writer.add_scalar('Loss/Validation', val_metrics.get('val_loss', 0), epoch)
        
        # Learning rate
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Position errors
        self.writer.add_scalar('Position_Error/Train', train_metrics.get('position_error', 0), epoch)
        self.writer.add_scalar('Position_Error/Validation', val_metrics.get('position_error', 0), epoch)
        
        # Altitude errors
        self.writer.add_scalar('Altitude_Error/Train', train_metrics.get('altitude_error', 0), epoch)
        self.writer.add_scalar('Altitude_Error/Validation', val_metrics.get('altitude_error', 0), epoch)
        
        # RMSE
        self.writer.add_scalar('RMSE/Train', train_metrics.get('rmse', 0), epoch)
        self.writer.add_scalar('RMSE/Validation', val_metrics.get('rmse', 0), epoch)
        
        # MAE
        self.writer.add_scalar('MAE/Train', train_metrics.get('mae', 0), epoch)
        self.writer.add_scalar('MAE/Validation', val_metrics.get('mae', 0), epoch)
        
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()
        logger.info("TensorBoard logger closed")
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
