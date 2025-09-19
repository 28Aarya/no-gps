<<<<<<< HEAD
import torch
import numpy as np

def expect_shape(x, spec, name="tensor"):
    """Validate tensor/array shape against specification.
    
    Args:
        x: Tensor or numpy array
        spec: Tuple with ints or None; None = wildcard
        name: Name for error messages
    
    Example:
        expect_shape(x, (None, 20, 7), "input")  # batch, seq_len=20, features=7
    """
    if x is None:
        raise ValueError(f"{name} is None")
    if len(x.shape) != len(spec):
        raise ValueError(f"{name} ndims {x.ndim} != expected {len(spec)}")
    for i, (a,e) in enumerate(zip(x.shape, spec)):
        if e is not None and a != e:
            raise ValueError(f"{name} dim {i}: got {a}, expected {e}")

def assert_finite(x, name="tensor"):
    """Check for NaN/Inf values in tensor/array."""
    if isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            raise ValueError(f"{name} contains {bad} non-finite values (NaN/Inf)")
    elif isinstance(x, np.ndarray):
        if not np.isfinite(x).all():
            bad = (~np.isfinite(x)).sum()
            raise ValueError(f"{name} contains {bad} non-finite values (NaN/Inf)")

def assert_dtype(x, dtype, name="tensor"):
    """Check tensor/array data type."""
    if x.dtype != dtype:
        raise TypeError(f"{name} dtype {x.dtype} != expected {dtype}")

def validate_sequence_batch(batch, expected_features=None, expected_targets=2):
    """Validate a batch of sequences with comprehensive checks.
    
    Args:
        batch: Dictionary with 'input_ids', 'labels', etc.
        expected_features: Expected number of features (7 for our data)
        expected_targets: Expected number of target dimensions (2 for east,north residuals)
    """
    X = batch['input_ids']
    y = batch['labels']
    
    # Basic shape validation
    expect_shape(X, (None, None, None), "batch_input")  # (batch, seq_len, features)
    expect_shape(y, (None, None, expected_targets), "batch_labels")  # (batch, pred_len, targets)
    
    # Data validation
    assert_finite(X, "batch_input")
    assert_finite(y, "batch_labels")
    assert_dtype(X, torch.float32, "batch_input")
    assert_dtype(y, torch.float32, "batch_labels")
    
    # Feature dimension validation
    if expected_features is not None:
        if X.shape[-1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[-1]}")
    
    return True

'''Use these at:

DataLoader sanity pass (first batch) - validate_sequence_batch()
Model input projector (after normalization) - expect_shape(), assert_finite()
Before/after attention layers (optional, at least first epoch) - expect_shape()
After feature projection - expect_shape(), assert_dtype()
=======
import torch
import numpy as np

def expect_shape(x, spec, name="tensor"):
    """Validate tensor/array shape against specification.
    
    Args:
        x: Tensor or numpy array
        spec: Tuple with ints or None; None = wildcard
        name: Name for error messages
    
    Example:
        expect_shape(x, (None, 20, 8), "input")  # batch, seq_len=20, features=8
    """
    if x is None:
        raise ValueError(f"{name} is None")
    if len(x.shape) != len(spec):
        raise ValueError(f"{name} ndims {x.ndim} != expected {len(spec)}")
    for i, (a,e) in enumerate(zip(x.shape, spec)):
        if e is not None and a != e:
            raise ValueError(f"{name} dim {i}: got {a}, expected {e}")

def assert_finite(x, name="tensor"):
    """Check for NaN/Inf values in tensor/array."""
    if isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            raise ValueError(f"{name} contains {bad} non-finite values (NaN/Inf)")
    elif isinstance(x, np.ndarray):
        if not np.isfinite(x).all():
            bad = (~np.isfinite(x)).sum()
            raise ValueError(f"{name} contains {bad} non-finite values (NaN/Inf)")

def assert_dtype(x, dtype, name="tensor"):
    """Check tensor/array data type."""
    if x.dtype != dtype:
        raise TypeError(f"{name} dtype {x.dtype} != expected {dtype}")

def validate_sequence_batch(batch, expected_features=None, expected_targets=3):
    """Validate a batch of sequences with comprehensive checks.
    
    Args:
        batch: Dictionary with 'input_ids', 'labels', etc.
        expected_features: Expected number of features (including ICAO ID)
        expected_targets: Expected number of target dimensions
    """
    X = batch['input_ids']
    y = batch['labels']
    
    # Basic shape validation
    expect_shape(X, (None, None, None), "batch_input")  # (batch, seq_len, features)
    expect_shape(y, (None, None, expected_targets), "batch_labels")  # (batch, pred_len, targets)
    
    # Data validation
    assert_finite(X, "batch_input")
    assert_finite(y, "batch_labels")
    assert_dtype(X, torch.float32, "batch_input")
    assert_dtype(y, torch.float32, "batch_labels")
    
    # Feature dimension validation
    if expected_features is not None:
        if X.shape[-1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[-1]}")
    
    # ICAO ID validation (should be in last feature position)
    if X.shape[-1] > 0:
        icao_features = X[:, :, -1]  # Last feature dimension
        for i in range(len(X)):
            unique_icaos = torch.unique(icao_features[i])
            if len(unique_icaos) != 1:
                raise ValueError(f"Sequence {i} has inconsistent ICAO IDs: {unique_icaos}")
    
    return True

'''Use these at:

DataLoader sanity pass (first batch) - validate_sequence_batch()
Model input projector (after normalization) - expect_shape(), assert_finite()
Before/after attention layers (optional, at least first epoch) - expect_shape()
After ICAO embedding (if used) - expect_shape(), assert_dtype()
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
'''