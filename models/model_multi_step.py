import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from typing import Dict, Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class AircraftTrajectoryTransformer(nn.Module):
    """
    Sequence-to-Sequence Transformer for Aircraft Trajectory Prediction
    Encoder: Processes full trajectory sequence (60 time steps)
    Decoder: Outputs multiple prediction points (lat, lon coordinates)
    """
    def __init__(self, 
                numerical_features_dim: int,
                icao_vocab_size: int,
                icao_embed_dim: int = 32,
                d_model: int = 128,
                num_heads: int = 8,
                num_layers: int = 6,
                d_ff: int = 512,
                max_seq_length: int = 100,
                dropout: float = 0.1,
                use_icao_embedding: bool = True,
                prediction_length: int = 10):
        super(AircraftTrajectoryTransformer, self).__init__()
        
        self.numerical_features_dim = numerical_features_dim
        self.icao_vocab_size = icao_vocab_size
        self.icao_embed_dim = icao_embed_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.use_icao_embedding = use_icao_embedding
        self.prediction_length = prediction_length
        
        # ICAO24 embedding layer
        if self.use_icao_embedding:
            self.icao_embedding = nn.Embedding(icao_vocab_size, icao_embed_dim)
            # Input projection for numerical features + ICAO embedding
            self.input_projection = nn.Linear(numerical_features_dim + icao_embed_dim, d_model)
        else:
            # Input projection for numerical features only
            self.input_projection = nn.Linear(numerical_features_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Decoder layers (for sequence-to-sequence)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Decoder input projection (multiple tokens for prediction)
        self.decoder_input_projection = nn.Linear(2, d_model)  # 2 for lat/lon
        
        # Output projection for prediction (lat, lon coordinates for multiple time steps)
        self.output_projection = nn.Linear(d_model, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights with smaller values to prevent NaN"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def generate_mask(self, src, tgt):
        """Generate masks for encoder and decoder"""
        # Source mask (encoder) - mask padding tokens
        src_mask = (src.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        
        # Target mask (decoder) - causal mask for multiple tokens
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=src.device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # (1,1,P,P)
        
        return src_mask, tgt_mask

    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                icao_id: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for sequence-to-sequence trajectory prediction
        
        Args:
            input_ids: Input sequence (batch_size, seq_len, numerical_features_dim)
            attention_mask: Attention mask (batch_size, seq_len)
            icao_id: ICAO24 IDs (batch_size,)
            
        Returns:
            Dictionary containing predictions and intermediate outputs
        """
        batch_size, seq_len, _ = input_ids.shape
        
        # Handle ICAO24 embeddings
        if self.use_icao_embedding and icao_id is not None:
            # Embed ICAO IDs
            icao_embedded = self.icao_embedding(icao_id)  # (batch_size, icao_embed_dim)
            
            # Expand ICAO embeddings to match sequence length
            icao_expanded = icao_embedded.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, icao_embed_dim)
            
            # Concatenate numerical features with ICAO embeddings
            combined_input = torch.cat([input_ids, icao_expanded], dim=-1)  # (batch_size, seq_len, numerical_features_dim + icao_embed_dim)
        else:
            combined_input = input_ids
        
        # Project to model dimension
        src_embedded = self.input_projection(combined_input)  # (batch_size, seq_len, d_model)
        src_embedded = self.dropout(self.positional_encoding(src_embedded))
        
        # Create decoder input (multiple tokens for prediction)
        # Use zeros as initial decoder input - the decoder will learn to predict from encoder context
        tgt = torch.zeros(batch_size, self.prediction_length, 2, device=input_ids.device)  # (batch_size, pred_len, 2)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_input_projection(tgt)))
        
        # Generate masks
        src_mask, tgt_mask = self.generate_mask(src_embedded, tgt_embedded)
        
        # Encoder: Process the full sequence
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        # Decoder: Process multiple tokens to get predictions
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # Output projection for prediction (lat, lon coordinates for multiple time steps)
        predictions = self.output_projection(dec_output)  # (batch_size, pred_len, 2)
        
        return {
            "predictions": predictions,
            "hidden_states": dec_output,
            "attention_weights": None  # Could be added if needed
        }

if __name__ == "__main__":
    # Test the model
    model = AircraftTrajectoryTransformer(
        numerical_features_dim=7,
        icao_vocab_size=1000,
        icao_embed_dim=32,
        d_model=128,
        num_heads=8,
        num_layers=6,
        max_seq_length=100,
        dropout=0.1,
        use_icao_embedding=True,
        prediction_length=10
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 60
    num_features = 7
    
    input_ids = torch.randn(batch_size, seq_len, num_features)
    icao_id = torch.randint(0, 1000, (batch_size,))
    
    outputs = model(input_ids=input_ids, icao_id=icao_id)
    print(f"Model output shape: {outputs['predictions'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    print(f"Expected output shape: (batch_size, prediction_length, 2) = ({batch_size}, 10, 2)")
    
