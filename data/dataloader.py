<<<<<<< HEAD
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import yaml
import json
import warnings
import math
from utils.shapes import expect_shape, assert_finite, assert_dtype  


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Dataset with configurable padding and truncation"""
    
    def __init__(self, 
                data_path: Union[str, Path],
                config: Dict[str, Any],
                split: str = "train"):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the data files
            config: Configuration dictionary
            split: Dataset split ("train", "val", "test")
        """
        self.config = config
        self.split = split
        self.data_path = Path(data_path)

        self.X, self.y = self._load_data()
        
        # Convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        
        logger.info(f"Loaded {split} dataset: X shape {self.X.shape}, y shape {self.y.shape}")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data arrays from NPZ files"""
        try:
            if self.split == "train":
                X = np.load(self.data_path / "trainX.npz", mmap_mode='r')["X"]
                y = np.load(self.data_path / "trainY.npz", mmap_mode='r')["y"]
            elif self.split == "val":
                X = np.load(self.data_path / "valX.npz", mmap_mode='r')["X"]
                y = np.load(self.data_path / "valY.npz", mmap_mode='r')["y"]
            elif self.split == "test":
                X = np.load(self.data_path / "testX.npz", mmap_mode='r')["X"]
                y = np.load(self.data_path / "testY.npz", mmap_mode='r')["y"]
            else:
                raise ValueError(f"Unknown split: {self.split}")

            logger.info(f"Loaded {self.split} data: X shape {X.shape}, y shape {y.shape}")
            
            # Basic validation
            if len(X) > 0:
                expect_shape(X, (None, None, None), f"{self.split}_X")
                expect_shape(y, (None, None, 3), f"{self.split}_y")
                assert_finite(X, f"{self.split}_X")
                assert_finite(y, f"{self.split}_y")

            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample with optional padding/truncation"""
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        
        # Handle padding and truncation based on config
        max_length = self.config['data'].get('max_length')
        truncation = self.config['data'].get('truncation', False)
        padding = self.config['data'].get('padding', False)
        
        if max_length is not None:
            if len(X_sample) > max_length:
                if truncation:
                    X_sample = X_sample[:max_length]
                    y_sample = y_sample[:self.config['model']['prediction_length']]
                elif padding:
                    # Pad with zeros
                    padding_length = max_length - len(X_sample)
                    padding_tensor = torch.zeros(padding_length, X_sample.shape[-1])
                    X_sample = torch.cat([X_sample, padding_tensor])
            elif len(X_sample) < max_length and padding:
                # Pad shorter sequences
                padding_length = max_length - len(X_sample)
                padding_tensor = torch.zeros(padding_length, X_sample.shape[-1])
                X_sample = torch.cat([X_sample, padding_tensor])
        
        # Shape validation for each sample
        expect_shape(X_sample, (None, None), "X_sample")  # (seq_len, features)
        expect_shape(y_sample, (None, 3), "y_sample")     # (pred_len, 3)
        assert_finite(X_sample, "X_sample")
        assert_finite(y_sample, "y_sample")      
        
        return X_sample, y_sample  # Return tuple as expected

class DataLoaderFactory:
    """Factory class for creating dataloaders"""
    
    @staticmethod
    def create_dataloaders(config: Dict[str, Any],
                        data_path: Optional[Union[str, Path]] = None) -> Dict[str, DataLoader]:
        """
        Create train, validation, and test dataloaders
        
        Args:
            config: Configuration dictionary
            data_path: Optional override for data path
            
        Returns:
            Dictionary containing train, val, and test dataloaders
        """
        if data_path is None:
            data_path = Path(config['data']['data_dir'])
        else:
            data_path = Path(data_path)
        
        # Validate data path exists
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Check for required files
        required_files = ["trainX.npz", "trainY.npz", "valX.npz", "valY.npz", "testX.npz", "testY.npz"]
        missing_files = [f for f in required_files if not (data_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files in {data_path}: {missing_files}")
        
        # Create datasets with error handling
        try:
            train_dataset = TimeSeriesDataset(data_path, config, split="train")
            val_dataset = TimeSeriesDataset(data_path, config, split="val")
            test_dataset = TimeSeriesDataset(data_path, config, split="test")
        except Exception as e:
            logger.error(f"Failed to create datasets: {e}")
            logger.error(f"Data path: {data_path}")
            logger.error(f"Expected files: trainX.npz, trainY.npz, valX.npz, valY.npz, testX.npz, testY.npz")
            raise

        # Get dataloader settings from config
        batch_size = config['data']['batch_size']
        num_workers = config['data']['num_workers']
        pin_memory = config['data']['pin_memory']
        shuffle = config['data']['shuffle']

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=config['data']['drop_last']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        logger.info(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }

def get_dataloaders(config: Dict[str, Any],
                data_path: Optional[Union[str, Path]] = None) -> Dict[str, DataLoader]:
    """
    Convenience function to get dataloaders
    
    Args:
        config: Configuration dictionary
        data_path: Optional override for data path
        
    Returns:
        Dictionary containing dataloaders
    """
    return DataLoaderFactory.create_dataloaders(config, data_path)
=======
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import yaml
import json
import warnings
import math
from utils.shapes import expect_shape, assert_finite, assert_dtype  


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Dataset with configurable padding and truncation"""
    
    def __init__(self, 
                data_path: Union[str, Path],
                config: Dict[str, Any],
                split: str = "train"):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the data files
            config: Configuration dictionary
            split: Dataset split ("train", "val", "test")
        """
        self.config = config
        self.split = split
        self.data_path = Path(data_path)

        self.X, self.y = self._load_data()
        
        # Convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        
        logger.info(f"Loaded {split} dataset: X shape {self.X.shape}, y shape {self.y.shape}")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data arrays from NPZ files"""
        try:
            if self.split == "train":
                X = np.load(self.data_path / "trainX.npz", mmap_mode='r')["X"]
                y = np.load(self.data_path / "trainY.npz", mmap_mode='r')["y"]
            elif self.split == "val":
                X = np.load(self.data_path / "valX.npz", mmap_mode='r')["X"]
                y = np.load(self.data_path / "valY.npz", mmap_mode='r')["y"]
            elif self.split == "test":
                X = np.load(self.data_path / "testX.npz", mmap_mode='r')["X"]
                y = np.load(self.data_path / "testY.npz", mmap_mode='r')["y"]
            else:
                raise ValueError(f"Unknown split: {self.split}")

            logger.info(f"Loaded {self.split} data: X shape {X.shape}, y shape {y.shape}")
            
            # Basic validation
            if len(X) > 0:
                expect_shape(X, (None, None, None), f"{self.split}_X")
                expect_shape(y, (None, None, 3), f"{self.split}_y")
                assert_finite(X, f"{self.split}_X")
                assert_finite(y, f"{self.split}_y")

            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample with optional padding/truncation"""
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        
        # Handle padding and truncation based on config
        max_length = self.config['data'].get('max_length')
        truncation = self.config['data'].get('truncation', False)
        padding = self.config['data'].get('padding', False)
        
        if max_length is not None:
            if len(X_sample) > max_length:
                if truncation:
                    X_sample = X_sample[:max_length]
                    y_sample = y_sample[:self.config['model']['prediction_length']]
                elif padding:
                    # Pad with zeros
                    padding_length = max_length - len(X_sample)
                    padding_tensor = torch.zeros(padding_length, X_sample.shape[-1])
                    X_sample = torch.cat([X_sample, padding_tensor])
            elif len(X_sample) < max_length and padding:
                # Pad shorter sequences
                padding_length = max_length - len(X_sample)
                padding_tensor = torch.zeros(padding_length, X_sample.shape[-1])
                X_sample = torch.cat([X_sample, padding_tensor])
        
        # Shape validation for each sample
        expect_shape(X_sample, (None, None), "X_sample")  # (seq_len, features)
        expect_shape(y_sample, (None, 3), "y_sample")     # (pred_len, 3)
        assert_finite(X_sample, "X_sample")
        assert_finite(y_sample, "y_sample")      
        
        return X_sample, y_sample  # Return tuple as expected

class DataLoaderFactory:
    """Factory class for creating dataloaders"""
    
    @staticmethod
    def create_dataloaders(config: Dict[str, Any],
                        data_path: Optional[Union[str, Path]] = None) -> Dict[str, DataLoader]:
        """
        Create train, validation, and test dataloaders
        
        Args:
            config: Configuration dictionary
            data_path: Optional override for data path
            
        Returns:
            Dictionary containing train, val, and test dataloaders
        """
        if data_path is None:
            data_path = Path(config['data']['data_dir'])
        else:
            data_path = Path(data_path)
        
        # Validate data path exists
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Check for required files
        required_files = ["trainX.npz", "trainY.npz", "valX.npz", "valY.npz", "testX.npz", "testY.npz"]
        missing_files = [f for f in required_files if not (data_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files in {data_path}: {missing_files}")
        
        # Create datasets with error handling
        try:
            train_dataset = TimeSeriesDataset(data_path, config, split="train")
            val_dataset = TimeSeriesDataset(data_path, config, split="val")
            test_dataset = TimeSeriesDataset(data_path, config, split="test")
        except Exception as e:
            logger.error(f"Failed to create datasets: {e}")
            logger.error(f"Data path: {data_path}")
            logger.error(f"Expected files: trainX.npz, trainY.npz, valX.npz, valY.npz, testX.npz, testY.npz")
            raise

        # Get dataloader settings from config
        batch_size = config['data']['batch_size']
        num_workers = config['data']['num_workers']
        pin_memory = config['data']['pin_memory']
        shuffle = config['data']['shuffle']

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=config['data']['drop_last']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        logger.info(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }

def get_dataloaders(config: Dict[str, Any],
                data_path: Optional[Union[str, Path]] = None) -> Dict[str, DataLoader]:
    """
    Convenience function to get dataloaders
    
    Args:
        config: Configuration dictionary
        data_path: Optional override for data path
        
    Returns:
        Dictionary containing dataloaders
    """
    return DataLoaderFactory.create_dataloaders(config, data_path)
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
