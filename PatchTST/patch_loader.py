<<<<<<< HEAD
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from .patch_dataset import PatchTSTDataset


def build_patch_dataloaders(
    *,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    group_ids_train=None,
    group_ids_val=None,
    group_ids_test=None,
    config: Dict[str, Any],
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for PatchTST using on-the-fly patching.
    Expects config.model: seq_len, pred_len, d_model, patch_len, and optional patch_stride, window_stride
            config.data: batch_size, num_workers, pin_memory, shuffle, drop_last
    """
    mcfg = config['model']
    dcfg = config['data']

    patch_stride = mcfg.get('patch_stride', mcfg['patch_len'])
    window_stride = mcfg.get('window_stride', patch_stride)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_ds = PatchTSTDataset(
        x=x_train, y=y_train, group_ids=group_ids_train,
        seq_len=mcfg['seq_len'], pred_len=mcfg['pred_len'],
        patch_len=mcfg['patch_len'], patch_stride=patch_stride,
        window_stride=window_stride, device=device,
    )
    val_ds = PatchTSTDataset(
        x=x_val, y=y_val, group_ids=group_ids_val,
        seq_len=mcfg['seq_len'], pred_len=mcfg['pred_len'],
        patch_len=mcfg['patch_len'], patch_stride=patch_stride,
        window_stride=window_stride, device=device,
    )
    test_ds = PatchTSTDataset(
        x=x_test, y=y_test, group_ids=group_ids_test,
        seq_len=mcfg['seq_len'], pred_len=mcfg['pred_len'],
        patch_len=mcfg['patch_len'], patch_stride=patch_stride,
        window_stride=window_stride, device=device,
    )

    loaders = {
        'train': DataLoader(train_ds, batch_size=dcfg['batch_size'], shuffle=dcfg['shuffle'],
                            num_workers=dcfg['num_workers'], pin_memory=dcfg['pin_memory'], drop_last=dcfg['drop_last']),
        'val': DataLoader(val_ds, batch_size=dcfg['batch_size'], shuffle=False,
                        num_workers=dcfg['num_workers'], pin_memory=dcfg['pin_memory'], drop_last=False),
        'test': DataLoader(test_ds, batch_size=dcfg['batch_size'], shuffle=False,
                        num_workers=dcfg['num_workers'], pin_memory=dcfg['pin_memory'], drop_last=False),
    }

    return loaders


=======
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from .patch_dataset import PatchTSTDataset


def build_patch_dataloaders(
    *,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    group_ids_train=None,
    group_ids_val=None,
    group_ids_test=None,
    config: Dict[str, Any],
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for PatchTST using on-the-fly patching.
    Expects config.model: seq_len, pred_len, d_model, patch_len, and optional patch_stride, window_stride
            config.data: batch_size, num_workers, pin_memory, shuffle, drop_last
    """
    mcfg = config['model']
    dcfg = config['data']

    patch_stride = mcfg.get('patch_stride', mcfg['patch_len'])
    window_stride = mcfg.get('window_stride', patch_stride)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_ds = PatchTSTDataset(
        x=x_train, y=y_train, group_ids=group_ids_train,
        seq_len=mcfg['seq_len'], pred_len=mcfg['pred_len'],
        patch_len=mcfg['patch_len'], patch_stride=patch_stride,
        window_stride=window_stride, device=device,
    )
    val_ds = PatchTSTDataset(
        x=x_val, y=y_val, group_ids=group_ids_val,
        seq_len=mcfg['seq_len'], pred_len=mcfg['pred_len'],
        patch_len=mcfg['patch_len'], patch_stride=patch_stride,
        window_stride=window_stride, device=device,
    )
    test_ds = PatchTSTDataset(
        x=x_test, y=y_test, group_ids=group_ids_test,
        seq_len=mcfg['seq_len'], pred_len=mcfg['pred_len'],
        patch_len=mcfg['patch_len'], patch_stride=patch_stride,
        window_stride=window_stride, device=device,
    )

    loaders = {
        'train': DataLoader(train_ds, batch_size=dcfg['batch_size'], shuffle=dcfg['shuffle'],
                            num_workers=dcfg['num_workers'], pin_memory=dcfg['pin_memory'], drop_last=dcfg['drop_last']),
        'val': DataLoader(val_ds, batch_size=dcfg['batch_size'], shuffle=False,
                        num_workers=dcfg['num_workers'], pin_memory=dcfg['pin_memory'], drop_last=False),
        'test': DataLoader(test_ds, batch_size=dcfg['batch_size'], shuffle=False,
                        num_workers=dcfg['num_workers'], pin_memory=dcfg['pin_memory'], drop_last=False),
    }

    return loaders


>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
