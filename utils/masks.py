import torch

def make_pad_mask(lengths, max_len):
    # lengths: (B,)
    rng = torch.arange(max_len, device=lengths.device)[None, :]
    return rng >= lengths[:, None]  # True on pads

def to_attn_mask_from_pad(pad_mask):
    # pad_mask: (B, T_kv) True=pad → attn_mask: (B,1,1,T_kv) with -inf on pads
    # convert this inside attention to additive mask on scores
    return pad_mask[:, None, None, :]

class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self.mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                        torch.arange(H)[None, :, None],
                        index, :]
        self.mask = indicator.view(scores.shape)
