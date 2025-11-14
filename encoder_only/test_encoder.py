import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoder_only.encoder import AircraftTrajectoryEncoder

def test_model():
    # Model parameters
    input_features = 7  # 7 continuous features: heading, vertrate, velocity, baroaltitude, east, north, up
    d_model = 64
    batch_size = 4
    seq_len = 20
    
    # Create model
    model = AircraftTrajectoryEncoder(
        input_features=input_features,
        d_model=d_model,
        num_heads=8,
        num_layers=6,
        prediction_length=20,
        target_dim=2  # Changed to 2 for east,north residuals
    )
    
    # Test input
    x = torch.randn(batch_size, seq_len, input_features)  # All features are continuous
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, output

if __name__ == "__main__":
    model, output = test_model()