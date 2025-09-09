<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from iTransformer.layers.Transformer_EncDec import Encoder, EncoderLayer
from iTransformer.layers.SelfAttention_Family import FullAttention, AttentionLayer
from iTransformer.layers.Embed import DataEmbedding_inverted
import numpy as np
from utils.shapes import expect_shape, assert_finite, assert_dtype
class AircraftiTransformer(nn.Module):
    
    def __init__(self, configs):
        super(AircraftiTransformer, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.output_attention = configs['output_attention']
        self.use_norm = configs['use_norm']
        

        self.numerical_features = configs['numerical_features']  # 8 features
        self.output_dim = configs['output_dim']  # 3 for lat, lon, altitude
        self.d_model = configs['d_model']
        
        # DataEmbedding_inverted with numerical features
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len,  # Use seq_len (20) as input dimension
            configs['d_model'], 
            configs['embed'], 
            configs['freq'],
            configs['dropout']
        )
        
        # Encoder stack
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs['factor'], 
                                attention_dropout=configs['dropout'],
                                output_attention=configs['output_attention']), 
                    configs['d_model'], configs['n_heads']),
                configs['d_model'],
                configs['d_ff'],
                dropout=configs['dropout'],
                activation=configs['activation']
            ) for l in range(configs['e_layers'])
        ], norm_layer=torch.nn.LayerNorm(configs['d_model']))

        # Output projection
        self.projector = nn.Linear(configs['d_model'], self.pred_len, bias=True)

        self.debug_shapes = configs.get('debug_shapes', False)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Shape validation
        expect_shape(x_enc, (None, self.seq_len, self.numerical_features), "x_enc")
        assert_finite(x_enc, "x_enc")
        
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev


        # Embed: [B, seq_len, features] -> [B, features, d_model]
        enc_out = self.enc_embedding(x_enc, None)
        if self.debug_shapes:
            print(f"[Shapes] enc_out: {enc_out.shape}")
        expect_shape(enc_out, (None, self.numerical_features, self.d_model), "enc_out")
        
        # Encode: [B, features, d_model] -> [B, features, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        if self.debug_shapes:
            print(f"[Shapes] enc_out_after_encoder: {enc_out.shape}")
        expect_shape(enc_out, (None, self.numerical_features, self.d_model), "enc_out_after_encoder")

        # Project: [B, features, d_model] -> [B, features, pred_len] -> [B, pred_len, features]
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        if self.debug_shapes:
            print(f"[Shapes] dec_out pre-trim: {dec_out.shape}")
        expect_shape(dec_out, (None, self.pred_len, self.numerical_features), "dec_out")
        
        # Take only output dimensions: [B, pred_len, output_dim]
        dec_out = dec_out[:, :, :self.output_dim]
        expect_shape(dec_out, (None, self.pred_len, self.output_dim), "final_output")

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :self.output_dim].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :self.output_dim].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [B, seq_len, features]
        expect_shape(x_enc, (None, self.seq_len, self.numerical_features), "x_enc")
        
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embed: [B, seq_len, features] -> [B, features, d_model]
        enc_out = self.enc_embedding(x_enc, None)
        expect_shape(enc_out, (None, self.numerical_features, self.d_model), "enc_out")
        
        # Encode: [B, features, d_model] -> [B, features, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Project: [B, features, d_model] -> [B, features, pred_len] -> [B, pred_len, features]
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        
        # Take only output dimensions: [B, pred_len, output_dim]
        dec_out = dec_out[:, :, :self.output_dim]
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :self.output_dim].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :self.output_dim].unsqueeze(1).repeat(1, self.pred_len, 1))

=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from iTransformer.layers.Transformer_EncDec import Encoder, EncoderLayer
from iTransformer.layers.SelfAttention_Family import FullAttention, AttentionLayer
from iTransformer.layers.Embed import DataEmbedding_inverted
import numpy as np
from utils.shapes import expect_shape, assert_finite, assert_dtype


class AircraftiTransformer(nn.Module):
    
    def __init__(self, configs):
        super(AircraftiTransformer, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.output_attention = configs['output_attention']
        self.use_norm = configs['use_norm']
        

        self.numerical_features = configs['numerical_features']  # 8 features
        self.output_dim = configs['output_dim']  # 3 for lat, lon, altitude
        self.d_model = configs['d_model']
        
        # DataEmbedding_inverted with numerical features
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len,  # Use seq_len (20) as input dimension
            configs['d_model'], 
            configs['embed'], 
            configs['freq'],
            configs['dropout']
        )
        
        # Encoder stack
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs['factor'], 
                                attention_dropout=configs['dropout'],
                                output_attention=configs['output_attention']), 
                    configs['d_model'], configs['n_heads']),
                configs['d_model'],
                configs['d_ff'],
                dropout=configs['dropout'],
                activation=configs['activation']
            ) for l in range(configs['e_layers'])
        ], norm_layer=torch.nn.LayerNorm(configs['d_model']))

        # Output projection
        self.projector = nn.Linear(configs['d_model'], self.pred_len, bias=True)

        self.debug_shapes = configs.get('debug_shapes', False)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Shape validation
        expect_shape(x_enc, (None, self.seq_len, self.numerical_features), "x_enc")
        assert_finite(x_enc, "x_enc")
        
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev


        # Embed: [B, seq_len, features] -> [B, features, d_model]
        enc_out = self.enc_embedding(x_enc, None)
        if self.debug_shapes:
            print(f"[Shapes] enc_out: {enc_out.shape}")
        expect_shape(enc_out, (None, self.numerical_features, self.d_model), "enc_out")
        
        # Encode: [B, features, d_model] -> [B, features, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        if self.debug_shapes:
            print(f"[Shapes] enc_out_after_encoder: {enc_out.shape}")
        expect_shape(enc_out, (None, self.numerical_features, self.d_model), "enc_out_after_encoder")

        # Project: [B, features, d_model] -> [B, features, pred_len] -> [B, pred_len, features]
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        if self.debug_shapes:
            print(f"[Shapes] dec_out pre-trim: {dec_out.shape}")
        expect_shape(dec_out, (None, self.pred_len, self.numerical_features), "dec_out")
        
        # Take only output dimensions: [B, pred_len, output_dim]
        dec_out = dec_out[:, :, :self.output_dim]
        expect_shape(dec_out, (None, self.pred_len, self.output_dim), "final_output")

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :self.output_dim].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :self.output_dim].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [B, seq_len, features]
        expect_shape(x_enc, (None, self.seq_len, self.numerical_features), "x_enc")
        
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embed: [B, seq_len, features] -> [B, features, d_model]
        enc_out = self.enc_embedding(x_enc, None)
        expect_shape(enc_out, (None, self.numerical_features, self.d_model), "enc_out")
        
        # Encode: [B, features, d_model] -> [B, features, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Project: [B, features, d_model] -> [B, features, pred_len] -> [B, pred_len, features]
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        
        # Take only output dimensions: [B, pred_len, output_dim]
        dec_out = dec_out[:, :, :self.output_dim]
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :self.output_dim].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :self.output_dim].unsqueeze(1).repeat(1, self.pred_len, 1))

>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
        return dec_out