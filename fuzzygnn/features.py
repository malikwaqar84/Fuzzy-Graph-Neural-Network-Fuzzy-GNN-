"""Window extraction and the features of Section 3.2 (Eqs. 3-7), coupling and activation."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def windows(x: np.ndarray, L: int, S: int) -> np.ndarray:
    """x [C, T] -> [nW, C, L] (a view, no copy)."""
    v = sliding_window_view(x, L, axis=1)[:, ::S, :]   # [C, nW, L]
    return np.transpose(v, (1, 0, 2))


def window_features(z: np.ndarray, x_uv: np.ndarray, cfg: dict, chunk: int = 256):
    """Compute per-window short-scale features, coupling, activation and artefact flags.

    z     : normalised signal [C, T] at cfg fs
    x_uv  : filtered signal in microvolts at the same rate (for artefact detection)
    returns dict with feat_s [nW, C, nB+1], corr [nW, C, C], act [nW, C], art [nW]
    """
    fs = cfg["data"]["fs"]
    L = int(round(cfg["protocol"]["window_s"] * fs))
    S = int(round(cfg["protocol"]["stride_s"] * fs))
    Wz, Wu = windows(z, L, S), windows(x_uv, L, S)
    nW, C = Wz.shape[0], Wz.shape[1]
    freqs = np.fft.rfftfreq(L, 1 / fs)
    bands = list(cfg["features"]["bands"].values())
    lo, hi = cfg["data"]["band"]
    in_band = (freqs >= lo) & (freqs <= hi)
    feat = np.zeros((nW, C, len(bands) + 1), np.float32)
    corr = np.zeros((nW, C, C), np.float16)
    act = np.zeros((nW, C), np.float32)
    art = np.zeros(nW, bool)
    for s in range(0, nW, chunk):
        w = np.asarray(Wz[s:s + chunk], dtype=np.float64)           # [n, C, L]
        P = np.abs(np.fft.rfft(w, axis=-1)) ** 2                     # power spectrum
        for bi, (b0, b1) in enumerate(bands):
            m = (freqs >= b0) & (freqs < b1) if b1 < hi else (freqs >= b0) & (freqs <= b1)
            feat[s:s + chunk, :, bi] = np.log(P[..., m].sum(-1) + 1e-8)            # Eq. 4 (log)
        Pb = P[..., in_band]
        p = Pb / (Pb.sum(-1, keepdims=True) + 1e-12)
        feat[s:s + chunk, :, -1] = -(p * np.log(p + 1e-12)).sum(-1)               # Eq. 5
        act[s:s + chunk] = np.log(Pb.sum(-1) + 1e-8)                              # log total energy
        wc = w - w.mean(-1, keepdims=True)
        wc /= (np.linalg.norm(wc, axis=-1, keepdims=True) + 1e-8)
        corr[s:s + chunk] = np.einsum("ncl,ndl->ncd", wc, wc).astype(np.float16)   # Pearson c_ij
        u = np.asarray(Wu[s:s + chunk])
        art[s:s + chunk] = (np.abs(u).max(axis=(1, 2)) > cfg["data"]["artefact_uv"]) | \
                           (u.std(axis=2).min(axis=1) < cfg["data"]["flat_uv"])
    return {"feat_s": feat, "corr": corr, "act": act, "art": art, "L": L, "S": S}


def long_scale(feat_s: np.ndarray, K: int) -> np.ndarray:
    """Causal mean and average slope over the last K windows (Eq. 7). feat_s [nW, C, D]."""
    nW = feat_s.shape[0]
    cs = np.cumsum(np.concatenate([np.zeros_like(feat_s[:1]), feat_s]), axis=0)
    idx = np.arange(nW)
    start = np.maximum(0, idx - K + 1)
    k = (idx - start + 1).astype(np.float32)[:, None, None]
    mean = (cs[idx + 1] - cs[start]) / k
    slope = np.where(k > 1, (feat_s[idx] - feat_s[start]) / np.maximum(k - 1, 1), 0.0)
    return np.concatenate([mean, slope], axis=-1).astype(np.float32)
