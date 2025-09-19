import argparse
import sys
from pathlib import Path
import logging
import torch
import yaml
from torch.utils.data import DataLoader
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from iTransformer.iTransformer import AircraftiTransformer
from data.aircraft_dataset_wrapper import AircraftiTransformerDataset
from train.itransformer_trainer import iTransformerTrainer

# Create output directories
output_dirs = ['output/logs', 'output/checkpoints', 'output/plots', 'output/tensorboard']
for dir_path in output_dirs:
    # Remove if it's a file
    if os.path.exists(dir_path) and os.path.isfile(dir_path):
        os.remove(dir_path)
    # Create directory
    os.makedirs(dir_path, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/logs/main.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config_path = "utils/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting iTransformer training...")
    
    # Create datasets
    data_path = Path(config['data']['data_dir'])
    train_dataset = AircraftiTransformerDataset(data_path, config, split="train")
    val_dataset = AircraftiTransformerDataset(data_path, config, split="val")
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # Add this debug code
    # Check first item from train dataset
    x_enc, x_mark_enc, x_dec, x_mark_dec, y = train_dataset[0]
    logger.info("\nFirst training sample shapes:")
    logger.info(f"x_enc: {x_enc.shape}")
    logger.info(f"x_mark_enc: {x_mark_enc.shape}")
    logger.info(f"x_dec: {x_dec.shape}")
    logger.info(f"x_mark_dec: {x_mark_dec.shape}")
    logger.info(f"y: {y.shape}")
    logger.info(f"Number of features: {x_enc.shape[-1]}\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=config['data']['shuffle'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=config['data']['drop_last']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=False
    )
    
    # Create model
    model = AircraftiTransformer(config['model'])
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer (using your existing components)
    trainer = iTransformerTrainer(model, config)
    
    # Start training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        auto_resume=config.get('auto_resume', True)
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()