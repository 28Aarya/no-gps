<<<<<<< HEAD
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PatchTST.patch_loader import build_patch_dataloaders
from PatchTST.patchtst import build_patchtst_model
from train.trainer import Trainer
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, 'utils', 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
from types import SimpleNamespace

def load_split(csv_path, feature_cols, target_cols, group_id_col):
    """Load and prepare one split with feature selection"""
    df = pd.read_csv(csv_path)
    
    # Extract group IDs for boundaries (not fed to model)
    group_ids = df[group_id_col].astype(str).values
    
    # Select only the features we want for X (patching)
    X = df[feature_cols].to_numpy(dtype=np.float32)     # (N, C)
    Y = df[target_cols].to_numpy(dtype=np.float32)      # (N, O)
    
    # Reshape for dataset: each row becomes (1, features) 
    # Dataset will group by group_id and flatten time within each flight
    X = X[:, None, :]   # (N, 1, C)
    Y = Y[:, None, :]   # (N, 1, O)
    
    return X, Y, group_ids

def set_experiment_name(config):
    """Set experiment name based on model config"""
    m = config['model']
    name = f"PatchTST_d{m['d_model']}_h{m['n_heads']}_L{m['e_layers']}_pl{m['patch_len']}_sl{m['seq_len']}_pred{m['pred_len']}"
    if m.get('fusion', False):
        name += "_fusion"
    config['experiment']['name'] = name
    print(f"Experiment: {name}")

if __name__ == "__main__":
    # Set experiment name first
    set_experiment_name(config)
    
    # Load your pre-split CSVs with feature selection
    Xtr, Ytr, Gtr = load_split("data/new/train.csv", 
                            config['data']['feature_cols'], 
                            config['data']['target_cols'], 
                            config['data']['group_id_col'])
    Xva, Yva, Gva = load_split("data/new/val.csv", 
                            config['data']['feature_cols'], 
                            config['data']['target_cols'], 
                            config['data']['group_id_col'])
    Xte, Yte, Gte = load_split("data/new/test.csv", 
                            config['data']['feature_cols'], 
                            config['data']['target_cols'], 
                            config['data']['group_id_col'])
    
    # Build dataloaders
    loaders = build_patch_dataloaders(
        x_train=Xtr, y_train=Ytr, x_val=Xva, y_val=Yva, x_test=Xte, y_test=Yte,
        group_ids_train=Gtr, group_ids_val=Gva, group_ids_test=Gte,
        config=config
    )
    
    # Build model
    mcfg = SimpleNamespace(**config['model'])
    model = build_patchtst_model(mcfg)
    
    # Train
    trainer = Trainer(model, config)
=======
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PatchTST.patch_loader import build_patch_dataloaders
from PatchTST.patchtst import build_patchtst_model
from train.trainer import Trainer
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, 'utils', 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
from types import SimpleNamespace

def load_split(csv_path, feature_cols, target_cols, group_id_col):
    """Load and prepare one split with feature selection"""
    df = pd.read_csv(csv_path)
    
    # Extract group IDs for boundaries (not fed to model)
    group_ids = df[group_id_col].astype(str).values
    
    # Select only the features we want for X (patching)
    X = df[feature_cols].to_numpy(dtype=np.float32)     # (N, C)
    Y = df[target_cols].to_numpy(dtype=np.float32)      # (N, O)
    
    # Reshape for dataset: each row becomes (1, features) 
    # Dataset will group by group_id and flatten time within each flight
    X = X[:, None, :]   # (N, 1, C)
    Y = Y[:, None, :]   # (N, 1, O)
    
    return X, Y, group_ids

def set_experiment_name(config):
    """Set experiment name based on model config"""
    m = config['model']
    name = f"PatchTST_d{m['d_model']}_h{m['n_heads']}_L{m['e_layers']}_pl{m['patch_len']}_sl{m['seq_len']}_pred{m['pred_len']}"
    if m.get('fusion', False):
        name += "_fusion"
    config['experiment']['name'] = name
    print(f"Experiment: {name}")

if __name__ == "__main__":
    # Set experiment name first
    set_experiment_name(config)
    
    # Load your pre-split CSVs with feature selection
    Xtr, Ytr, Gtr = load_split("data/new/train.csv", 
                            config['data']['feature_cols'], 
                            config['data']['target_cols'], 
                            config['data']['group_id_col'])
    Xva, Yva, Gva = load_split("data/new/val.csv", 
                            config['data']['feature_cols'], 
                            config['data']['target_cols'], 
                            config['data']['group_id_col'])
    Xte, Yte, Gte = load_split("data/new/test.csv", 
                            config['data']['feature_cols'], 
                            config['data']['target_cols'], 
                            config['data']['group_id_col'])
    
    # Build dataloaders
    loaders = build_patch_dataloaders(
        x_train=Xtr, y_train=Ytr, x_val=Xva, y_val=Yva, x_test=Xte, y_test=Yte,
        group_ids_train=Gtr, group_ids_val=Gva, group_ids_test=Gte,
        config=config
    )
    
    # Build model
    mcfg = SimpleNamespace(**config['model'])
    model = build_patchtst_model(mcfg)
    
    # Train
    trainer = Trainer(model, config)
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
    trainer.train(loaders['train'], loaders['val'], num_epochs=config['num_epochs'])