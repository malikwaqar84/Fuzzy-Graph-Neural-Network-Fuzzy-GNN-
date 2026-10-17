"""Compact PyTorch re-implementations of the nine baseline methods (Section 4.6).

These follow the architectures described in the original papers but are simplified
so that every model uses the same inputs, windows, folds, loss and alarm rule.
The simplifications are listed in the README and in the docstring of each class.
Every model maps a batch to logits of shape [B, T].
"""
from __future__ import annotations

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import FuzzyAttentionLayer


def _flat(batch):
    raw = batch["raw"]                       # [B, T, C, L]
    B, T, C, L = raw.shape
    return raw.reshape(B * T, C, L), B, T


# 1 ----------------------------------------------------------------------------
class TruongCNN(nn.Module):
    """Truong et al. (2018): STFT spectrogram + CNN. Simplified: 2D convolutions with EEG
    channels as input planes instead of the original 3D first layer."""
    needs_raw = True

    def __init__(self, cfg, C, **_):
        super().__init__()
        self.fs = cfg["data"]["fs"]
        self.register_buffer("win", torch.hann_window(self.fs))
        blocks, d = [], C
        for k in (16, 32, 64):
            blocks += [nn.Conv2d(d, k, 3, padding=1), nn.BatchNorm2d(k), nn.ReLU(), nn.MaxPool2d(2)]
            d = k
        self.cnn = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(64 * 4, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, batch):
        x, B, T = _flat(batch)
        M, C, L = x.shape
        S = torch.stft(x.reshape(M * C, L), n_fft=self.fs, hop_length=self.fs // 4, window=self.win,
                       return_complex=True)
        S = torch.log(S.abs()[:, 1:41] + 1e-6).reshape(M, C, 40, -1)     # 1-40 Hz
        return self.fc(self.cnn(S)).view(B, T)


# 2 ----------------------------------------------------------------------------
class TsiourisLSTM(nn.Module):
    """Tsiouris et al. (2018): two-layer LSTM over sequences of handcrafted features.
    Simplified: uses our short-scale spectral features instead of the original feature set."""
    needs_raw = False

    def __init__(self, cfg, C, d_short, **_):
        super().__init__()
        self.d_short = d_short
        self.lstm = nn.LSTM(C * d_short, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, batch):
        x = batch["feat"][..., :self.d_short]
        B, T, C, D = x.shape
        h, _ = self.lstm(x.reshape(B, T, C * D))
        return self.fc(h).squeeze(-1)


# 3 ----------------------------------------------------------------------------
class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., 2018)."""
    needs_raw = True

    def __init__(self, cfg, C, L, F1=8, D=2, F2=16, **_):
        super().__init__()
        k = cfg["data"]["fs"] // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.25), nn.Flatten())
        with torch.no_grad():
            n = self.net(torch.zeros(1, 1, C, L)).shape[1]
        self.fc = nn.Linear(n, 1)

    def forward(self, batch):
        x, B, T = _flat(batch)
        return self.fc(self.net(x.unsqueeze(1))).view(B, T)


# 4 ----------------------------------------------------------------------------
class EEGConformer(nn.Module):
    """EEG Conformer (Song et al., 2023): convolutional tokeniser + transformer encoder."""
    needs_raw = True

    def __init__(self, cfg, C, L, emb=40, depth=2, heads=4, **_):
        super().__init__()
        self.tok = nn.Sequential(
            nn.Conv2d(1, emb, (1, 25)), nn.Conv2d(emb, emb, (C, 1)), nn.BatchNorm2d(emb), nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)), nn.Dropout(0.5))
        layer = nn.TransformerEncoderLayer(emb, heads, emb * 4, 0.5, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.fc = nn.Sequential(nn.Linear(emb, 32), nn.ELU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, batch):
        x, B, T = _flat(batch)
        t = self.tok(x.unsqueeze(1)).squeeze(2).transpose(1, 2)      # [M, tokens, emb]
        return self.fc(self.enc(t).mean(1)).view(B, T)


# wavelet helper ---------------------------------------------------------------
class DWT(nn.Module):
    """Multi-level discrete wavelet transform implemented with strided convolutions."""

    def __init__(self, wavelet="db4", levels=4):
        super().__init__()
        w = pywt.Wavelet(wavelet)
        self.register_buffer("lo", torch.tensor(w.dec_lo[::-1], dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("hi", torch.tensor(w.dec_hi[::-1], dtype=torch.float32).view(1, 1, -1))
        self.levels = levels

    def forward(self, x):                                   # x [M, C, L]
        M, C, L = x.shape
        a = x.reshape(M * C, 1, L)
        out = []
        p = self.lo.shape[-1] - 1
        for _ in range(self.levels):
            ap = F.pad(a, (p // 2, p - p // 2), mode="reflect")
            d = F.conv1d(ap, self.hi, stride=2)
            a = F.conv1d(ap, self.lo, stride=2)
            out.append(d.reshape(M, C, -1))
        out.append(a.reshape(M, C, -1))
        return out                                           # [D1, D2, ..., Dn, An]


# 5 ----------------------------------------------------------------------------
class AttentionWaveletCNN(nn.Module):
    """Xin et al. (2022): multi-scale wavelet decomposition, CNN branches and attention.
    Simplified: one 1D-CNN branch per sub-band followed by squeeze-and-excitation attention."""
    needs_raw = True

    def __init__(self, cfg, C, **_):
        super().__init__()
        self.dwt = DWT("db4", 4)
        self.branches = nn.ModuleList([nn.Sequential(
            nn.Conv1d(C, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()) for _ in range(5)])
        self.se = nn.Sequential(nn.Linear(160, 40), nn.ReLU(), nn.Linear(40, 160), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(160, 1))

    def forward(self, batch):
        x, B, T = _flat(batch)
        z = torch.cat([br(s) for br, s in zip(self.branches, self.dwt(x))], 1)
        return self.fc(z * self.se(z)).view(B, T)


# 6 ----------------------------------------------------------------------------
class WTRPNet(nn.Module):
    """Xin et al. (2021): recurrence plots in the wavelet domain fed to a CNN.
    Simplified: unthresholded recurrence plot of the level-3 approximation of each channel."""
    needs_raw = True

    def __init__(self, cfg, C, **_):
        super().__init__()
        self.dwt = DWT("db4", 3)
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, batch):
        x, B, T = _flat(batch)
        a = self.dwt(x)[-1]                                     # [M, C, ~80]
        rp = (a.unsqueeze(-1) - a.unsqueeze(-2)).abs()          # recurrence plot per channel
        rp = rp / (rp.amax(dim=(-1, -2), keepdim=True) + 1e-6)
        return self.fc(self.cnn(rp)).view(B, T)


# 7 ----------------------------------------------------------------------------
class ChenGCNGRU(nn.Module):
    """Chen et al. (2021): correlation graph + GCN for spatial features, GRU over time."""
    needs_raw = False

    def __init__(self, cfg, C, d_short, hidden=64, **_):
        super().__init__()
        self.d_short = d_short
        self.g1, self.g2 = nn.Linear(d_short, hidden), nn.Linear(hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, batch):
        x = batch["feat"][..., :self.d_short]
        B, T, C, D = x.shape
        A = batch["corr"].abs().reshape(B * T, C, C)
        d = A.sum(-1).clamp(min=1e-6).pow(-0.5)
        An = d.unsqueeze(-1) * A * d.unsqueeze(-2)
        h = F.relu(torch.bmm(An, self.g1(x.reshape(B * T, C, D))))
        h = F.relu(torch.bmm(An, self.g2(h))).mean(1).view(B, T, -1)
        o, _ = self.gru(h)
        return self.fc(o).squeeze(-1)


# 8 ----------------------------------------------------------------------------
class DCGRUCell(nn.Module):
    def __init__(self, d_in, hidden, K=2):
        super().__init__()
        self.K = K
        self.gates = nn.Linear((d_in + hidden) * (K + 1), 2 * hidden)
        self.cand = nn.Linear((d_in + hidden) * (K + 1), hidden)

    def diffuse(self, x, P):
        out, xk = [x], x
        for _ in range(self.K):
            xk = torch.bmm(P, xk)
            out.append(xk)
        return torch.cat(out, -1)

    def forward(self, x, h, P):
        zr = torch.sigmoid(self.gates(self.diffuse(torch.cat([x, h], -1), P)))
        z, r = zr.chunk(2, -1)
        c = torch.tanh(self.cand(self.diffuse(torch.cat([x, r * h], -1), P)))
        return z * h + (1 - z) * c


class TangDCRNN(nn.Module):
    """Tang et al. (2022): diffusion convolutional recurrent network on a correlation graph
    with top-k edges. Simplified: no self-supervised pre-training."""
    needs_raw = False

    def __init__(self, cfg, C, d_short, hidden=64, top_k=3, **_):
        super().__init__()
        self.d_short, self.hidden, self.k = d_short, hidden, top_k
        self.cell = DCGRUCell(d_short, hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, batch):
        x = batch["feat"][..., :self.d_short]
        B, T, C, D = x.shape
        A = batch["corr"].abs()
        eye = torch.eye(C, device=A.device)
        A = A * (1 - eye)
        thr = A.topk(self.k, dim=-1).values[..., -1:]
        A = (A >= thr).float() * A + eye
        P = A / A.sum(-1, keepdim=True)                     # random-walk transition matrix
        h = torch.zeros(B, C, self.hidden, device=x.device)
        outs = []
        for t in range(T):
            h = self.cell(x[:, t], h, P[:, t])
            outs.append(self.fc(h.mean(1)))
        return torch.cat(outs, -1)


# 9 ----------------------------------------------------------------------------
class DTSGAN(nn.Module):
    """Yan et al. (2025): LSTM temporal encoder with a dynamic graph attention network whose
    connectivity uses a Gaussian kernel. Simplified: one attention layer, Gaussian kernel on |c_ij|."""
    needs_raw = False

    def __init__(self, cfg, C, d_short, hidden=32, heads=4, **_):
        super().__init__()
        self.d_short = d_short
        self.lstm = nn.LSTM(d_short, hidden * heads, batch_first=True)
        self.gat = FuzzyAttentionLayer(hidden * heads, hidden, heads, 0.2)
        self.fc = nn.Linear(hidden * heads, 1)

    def forward(self, batch):
        x = batch["feat"][..., :self.d_short]
        B, T, C, D = x.shape
        h, _ = self.lstm(x.permute(0, 2, 1, 3).reshape(B * C, T, D))
        h = h.view(B, C, T, -1).permute(0, 2, 1, 3).reshape(B * T, C, -1)
        a = batch["corr"].abs().reshape(B * T, C, C)
        mu = torch.exp(-(1 - a) ** 2 / (2 * 0.25 ** 2))
        g = self.gat(h, mu).mean(1)
        return self.fc(g).view(B, T)


BASELINES = {
    "truong_cnn": TruongCNN, "tsiouris_lstm": TsiourisLSTM, "eegnet": EEGNet,
    "eeg_conformer": EEGConformer, "attention_wavelet_cnn": AttentionWaveletCNN, "wtrpnet": WTRPNet,
    "chen_gcn_gru": ChenGCNGRU, "tang_dcrnn": TangDCRNN, "dts_gan": DTSGAN,
}
