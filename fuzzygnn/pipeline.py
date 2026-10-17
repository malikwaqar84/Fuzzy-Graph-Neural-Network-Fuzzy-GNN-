"""Turning one recording (array or EDF file) into the cached features used by dataset.PatientData.

Used by scripts/prepare_data.py for CHB-MIT and by tests/make_synthetic.py, so both go
through exactly the same filtering, normalisation and feature code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import window_features
from .preprocess import filter_resample, read_edf, zscore_per_channel


def process_array(x_uv: np.ndarray, fs_in: float, cfg: dict):
    """x_uv [C, T] in microvolts at fs_in -> (feature dict, normalised signal z [C, T'])."""
    xf = filter_resample(x_uv, fs_in, cfg)          # filtered, resampled, still in microvolts
    z = zscore_per_channel(xf)                      # Eq. 2
    feats = window_features(z, xf, cfg)             # Eqs. 3-7, coupling, activation, artefacts
    return feats, z


def save_processed(out_dir: str | Path, stem: str, feats: dict, z: np.ndarray, save_raw: bool = True):
    """Write <stem>.npz (features) and, optionally, <stem>_z.npy (signal for raw-input baselines)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (stem + ".tmp.npz")
    np.savez(tmp, feat_s=feats["feat_s"], corr=feats["corr"], act=feats["act"],
             art=feats["art"], L=feats["L"], S=feats["S"])
    if save_raw:
        zt = out_dir / (stem + "_z.tmp.npy")
        np.save(zt, z.astype(np.float16))
        zt.replace(out_dir / (stem + "_z.npy"))
    tmp.replace(out_dir / (stem + ".npz"))          # written last: its presence means "done"


def process_edf(edf_path: str | Path, out_dir: str | Path, cfg: dict, save_raw: bool = True):
    """Process one EDF file. Returns (status, message)."""
    edf_path = Path(edf_path)
    stem = edf_path.stem
    if (Path(out_dir) / (stem + ".npz")).exists():
        return "cached", ""
    data, fs = read_edf(str(edf_path), cfg["data"]["channels"])
    if data is None:
        return "skipped", fs                        # fs holds the reason here
    if abs(fs - cfg["data"]["fs_in"]) > 1e-6:
        return "skipped", f"sampling rate {fs} Hz differs from {cfg['data']['fs_in']} Hz"
    L = int(round(cfg["protocol"]["window_s"] * cfg["data"]["fs"]))
    if data.shape[1] / fs * cfg["data"]["fs"] < L:
        return "skipped", "recording shorter than one window"
    feats, z = process_array(data, fs, cfg)
    save_processed(out_dir, stem, feats, z, save_raw)
    return "ok", f"{feats['feat_s'].shape[0]} windows"
