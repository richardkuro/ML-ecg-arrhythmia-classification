"""
models.py
---------
All architectures in one file.

Models
------
1. ResNet1D        – strong 1D CNN baseline (Fazeli 2018)
2. TCN             – Temporal Convolutional Network
3. TransformerECG  – Encoder-only Transformer
4. KAN             – Kolmogorov-Arnold Network (1D signal)
5. ConvKAN         – CNN feature extractor → KAN classifier
6. S4ECG           – Simplified S4-style State Space Model
7. LiquidNet       – Liquid Time-Constant (LTC) Network

All models accept input of shape (B, 1, 187) and output (B, num_classes).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  1. ResNet1D  (baseline)
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, downsample=False):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (downsample or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class ResNet1D(nn.Module):
    """
    4-block 1D ResNet, ~100k params.
    Reference: Fazeli et al. 2018 (arXiv:1805.00794)
    """
    def __init__(self, num_classes: int = 5, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock1D(32,  64,  downsample=True),
            ResBlock1D(64,  128, downsample=True),
            ResBlock1D(128, 256, downsample=True),
            ResBlock1D(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
#  2. TCN  (Temporal Convolutional Network)
# ══════════════════════════════════════════════════════════════════════════════

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel,
                                           padding=pad, dilation=dilation)),
            nn.ReLU(), nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel,
                                           padding=pad, dilation=dilation)),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()
        self._pad = pad

    def forward(self, x):
        out = self.net(x)
        # causal: chop the future padding
        out = out[:, :, :x.size(2)]
        return self.relu(out + self.skip(x))


class TCN(nn.Module):
    """
    Dilated causal TCN. Each layer doubles the dilation, giving
    exponentially growing receptive field.
    """
    def __init__(self, num_classes: int = 5, in_channels: int = 1,
                 channels: int = 64, levels: int = 6, kernel: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        for i in range(levels):
            dil  = 2 ** i
            ic   = in_channels if i == 0 else channels
            layers.append(TCNBlock(ic, channels, kernel, dil, dropout))
        self.network = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
#  3. TransformerECG
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, T, D)
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerECG(nn.Module):
    """
    Patch-based Transformer: split 187-length signal into patches,
    project each patch to d_model, apply Transformer encoder, classify CLS token.
    """
    def __init__(self, num_classes: int = 5, seq_len: int = 187,
                 patch_size: int = 11, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        # number of patches (may pad)
        self.patch_size = patch_size
        n_patches = math.ceil(seq_len / patch_size)
        self.proj = nn.Linear(patch_size, d_model)
        self.cls  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pe   = PositionalEncoding(d_model, max_len=n_patches + 1, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):   # x: (B, 1, 187)
        x = x.squeeze(1)    # (B, 187)
        B, L = x.shape
        # pad to multiple of patch_size
        pad = (-L) % self.patch_size
        if pad:
            x = F.pad(x, (0, pad))
        x = x.reshape(B, -1, self.patch_size)   # (B, n_patches, patch_size)
        x = self.proj(x)                         # (B, n_patches, d_model)
        cls = self.cls.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)         # (B, 1+n_patches, d_model)
        x   = self.pe(x)
        x   = self.transformer(x)
        x   = self.norm(x[:, 0])                 # CLS token
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
#  4. KAN  (Kolmogorov-Arnold Network)
# ══════════════════════════════════════════════════════════════════════════════

class KANLayer(nn.Module):
    """
    Single KAN layer: each connection has a trainable spline activation.
    Uses B-spline basis (degree 3, grid_size buckets).
    Input:  (B, in_features)
    Output: (B, out_features)
    """
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_f  = in_features
        self.out_f = out_features
        self.k     = spline_order
        self.G     = grid_size

        # Grid: fixed, evenly spaced on [-1, 1]
        grid = torch.linspace(-1.0, 1.0, grid_size + 1)
        # Extend by spline_order on each side for proper boundary handling
        grid = torch.cat([
            grid[0].unsqueeze(0).expand(spline_order) - torch.arange(spline_order, 0, -1) * (2.0 / grid_size),
            grid,
            grid[-1].unsqueeze(0).expand(spline_order) + torch.arange(1, spline_order + 1) * (2.0 / grid_size),
        ])
        self.register_buffer("grid", grid)  # (G + 2k + 1,)

        n_basis = grid_size + spline_order  # number of B-spline basis functions
        # Spline coefficients: one per (in, out, basis)
        self.coeff = nn.Parameter(torch.randn(out_features, in_features, n_basis) * 0.1)
        # Residual linear (like SiLU shortcut in original KAN paper)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.base_bias   = nn.Parameter(torch.zeros(out_features))
        self.scale       = nn.Parameter(torch.ones(out_features, in_features))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis values.
        x:      (B, in_features)
        return: (B, in_features, n_basis)
        """
        x = x.unsqueeze(-1)             # (B, in, 1)
        grid = self.grid                 # (G + 2k + 1,)

        # De Boor recursion (degree 0)
        basis = ((x >= grid[:-1]) & (x < grid[1:])).float()  # (B, in, G+2k)

        for d in range(1, self.k + 1):
            left  = grid[:-(d + 1)]
            right = grid[d:-1]
            denom_l = right - left
            denom_l = torch.where(denom_l == 0, torch.ones_like(denom_l), denom_l)

            left2  = grid[1:-d]
            right2 = grid[d + 1:]
            denom_r = right2 - left2
            denom_r = torch.where(denom_r == 0, torch.ones_like(denom_r), denom_r)

            t1 = (x - left)  / denom_l  * basis[..., :-1]
            t2 = (right2 - x) / denom_r * basis[..., 1:]
            basis = t1 + t2

        return basis  # (B, in, n_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp to grid range for safety
        x = x.clamp(self.grid[self.k].item(), self.grid[-(self.k + 1)].item())

        basis  = self.b_splines(x)                          # (B, in, n_basis)
        # Spline output: einsum over in and n_basis
        spline = torch.einsum("bin,oin->bo", basis, self.coeff * self.scale.unsqueeze(-1))
        # Residual (base) activation
        base   = F.linear(F.silu(x), self.base_weight, self.base_bias)
        return spline + base


class KAN(nn.Module):
    """
    KAN for 1D ECG classification.
    Flattens the 187-length signal and passes through stacked KAN layers.
    """
    def __init__(self, num_classes: int = 5, seq_len: int = 187,
                 hidden_sizes: list = None, grid_size: int = 5):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        dims   = [seq_len] + hidden_sizes + [num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i + 1], grid_size=grid_size))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
        self.net = nn.ModuleList(layers)

    def forward(self, x):          # (B, 1, 187)
        x = x.squeeze(1)           # (B, 187)
        for layer in self.net:
            x = layer(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  5. ConvKAN  (CNN extractor + KAN classifier)
# ══════════════════════════════════════════════════════════════════════════════

class ConvKAN(nn.Module):
    """
    Multi-scale 1D CNN to extract features, then a KAN layer to classify.
    This is the hybrid most likely to outperform plain KAN on ECG.
    """
    def __init__(self, num_classes: int = 5, in_channels: int = 1,
                 feat_dim: int = 128, grid_size: int = 5):
        super().__init__()
        # Parallel convolutions at 3 scales
        self.branch3  = self._conv(in_channels, 32, 3)
        self.branch7  = self._conv(in_channels, 32, 7)
        self.branch15 = self._conv(in_channels, 32, 15)

        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.norm  = nn.LayerNorm(96)
        self.kan   = KANLayer(96, feat_dim, grid_size=grid_size)
        self.head  = nn.Linear(feat_dim, num_classes)

    @staticmethod
    def _conv(in_ch, out_ch, k):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )

    def forward(self, x):
        b3  = self.pool(self.branch3(x)).squeeze(-1)
        b7  = self.pool(self.branch7(x)).squeeze(-1)
        b15 = self.pool(self.branch15(x)).squeeze(-1)
        feat = torch.cat([b3, b7, b15], dim=1)   # (B, 96)
        feat = self.norm(feat)
        feat = self.kan(feat)
        return self.head(feat)


# ══════════════════════════════════════════════════════════════════════════════
#  6. S4ECG  (Simplified S4 State Space Model)
# ══════════════════════════════════════════════════════════════════════════════

class S4Layer(nn.Module):
    """
    Diagonal-plus-low-rank (DPLR) S4 — simplified diagonal-only version (S4D).
    Operates in frequency domain via FFT convolution.

    Discretisation: ZOH  A_bar = exp(dt*A),  B_bar = (A_bar - I) * A^-1 * B
    For diagonal A: reduces to element-wise ops.
    """
    def __init__(self, d_model: int = 64, seq_len: int = 187, dropout: float = 0.1):
        super().__init__()
        self.d  = d_model
        self.L  = seq_len

        # S4D-Real: A diagonal on the real line (negative for stability)
        log_A = torch.linspace(0, math.log(seq_len), d_model)
        self.log_A = nn.Parameter(-log_A)           # learnable, kept negative via exp
        self.B     = nn.Parameter(torch.randn(d_model) * 0.01)
        self.C     = nn.Parameter(torch.randn(d_model) * 0.01)
        self.D     = nn.Parameter(torch.ones(d_model))  # skip connection
        self.log_dt = nn.Parameter(torch.zeros(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)

    def _get_kernel(self, L: int) -> torch.Tensor:
        """Compute convolution kernel K of length L."""
        dt = torch.exp(self.log_dt).unsqueeze(-1).float()
        A  = -torch.exp(self.log_A).unsqueeze(-1).float()
        B  = self.B.unsqueeze(-1).float()
        C  = self.C.unsqueeze(-1).float()

        # Discretise (ZOH)
        # A_bar = exp(dt * A),  B_bar = (A_bar - 1) / A * B
        A_bar = torch.exp(dt * A)                      # (d, 1)
        B_bar = (A_bar - 1.0) / A * B * dt            # (d, 1)

        # Compute K[l] = C * A_bar^l * B_bar for l = 0..L-1
        steps = torch.arange(L, device=A.device).float()  # (L,)
        powers = A_bar ** steps.unsqueeze(0)               # (d, L)
        K = (C * B_bar * powers).real                       # (d, L)
        return K   # (d, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) — sequence-first
        B, L, D = x.shape
        K = self._get_kernel(L)              # (D, L)

        # FFT convolution along sequence dimension
        # Pad to length 2L for linear (non-circular) convolution
        # Force float32: cuFFT on Windows requires power-of-two sizes for fp16
        x_t = x.permute(0, 2, 1).float()    # (B, D, L)
        K   = K.float()
        fft_len = 2 * L
        Xf = torch.fft.rfft(x_t, n=fft_len)
        Kf = torch.fft.rfft(K,   n=fft_len)
        Yf = Xf * Kf.unsqueeze(0)
        y  = torch.fft.irfft(Yf, n=fft_len)[..., :L]  # (B, D, L)
        y  = y.to(x.dtype)                  # cast back to original dtype

        # Skip connection with D parameter
        y = y + self.D.view(1, D, 1) * x_t
        y = y.permute(0, 2, 1)              # (B, L, D)
        y = self.drop(self.out_proj(y))
        return self.norm(y + x)             # residual


class S4ECG(nn.Module):
    """
    Stack of S4 layers for ECG classification.
    Input: (B, 1, 187)  Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 5, seq_len: int = 187,
                 d_model: int = 64, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.layers     = nn.ModuleList([
            S4Layer(d_model, seq_len, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):           # (B, 1, 187)
        x = x.permute(0, 2, 1)     # (B, 187, 1)
        x = self.input_proj(x)     # (B, 187, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x.mean(dim=1))   # global average over time
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
#  7. BiGRU
# ══════════════════════════════════════════════════════════════════════════════

class BiGRU(nn.Module):
    """
    Bidirectional GRU — fast CUDA-optimized recurrent baseline.
    Processes 187 timesteps as a sequence, ~80k params.
    """
    def __init__(self, num_classes: int = 5, input_size: int = 1,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x):              # (B, 1, 187)
        x = x.permute(0, 2, 1)        # (B, 187, 1)
        out, _ = self.gru(x)          # (B, 187, hidden*2)
        out = out.mean(dim=1)         # global average over time
        return self.head(out)


# ══════════════════════════════════════════════════════════════════════════════
#  Registry
# ══════════════════════════════════════════════════════════════════════════════

def get_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    registry = {
        "resnet1d":     ResNet1D,
        "tcn":          TCN,
        "transformer":  TransformerECG,
        "kan":          KAN,
        "convkan":      ConvKAN,
        "s4":           S4ECG,
        "bigru":    BiGRU,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name](num_classes=num_classes, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_dummy = torch.randn(4, 1, 187).to(device)

    for name in ["resnet1d", "tcn", "transformer", "kan", "convkan", "s4", "liquidnet"]:
        model = get_model(name, num_classes=5).to(device)
        out   = model(x_dummy)
        print(f"{name:<14}  params={count_params(model):>7,}  out={tuple(out.shape)}")
