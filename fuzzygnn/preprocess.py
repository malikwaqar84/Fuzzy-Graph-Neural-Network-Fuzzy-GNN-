"""Reading, filtering, resampling and normalising CHB-MIT EDF recordings."""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def read_edf(path: str, channels: list[str]):
    """Return (data_uV [C, T], fs) for the requested bipolar channels, or (None, reason)."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    names = raw.ch_names
    # mne appends -0, -1 ... to duplicated channel names (e.g. T8-P8-0, T8-P8-1)
    lookup = {}
    for i, n in enumerate(names):
        key = n.upper()
        for suffix in ("-0", "-1", "-2"):
            if key.endswith(suffix) and key[:-2].count("-") == 1:
                key = key[:-2]
        lookup.setdefault(key, i)
    idx = []
    for ch in channels:
        if ch.upper() not in lookup:
            return None, f"missing channel {ch}"
        idx.append(lookup[ch.upper()])
    data = raw.get_data(picks=idx) * 1e6      # volts -> microvolts
    return data.astype(np.float32), float(raw.info["sfreq"])


def filter_resample(x_uv: np.ndarray, fs_in: float, cfg: dict) -> np.ndarray:
    """Notch, band-pass (zero-phase FIR) and resample to cfg['data']['fs']."""
    import mne
    d = cfg["data"]
    x = x_uv.astype(np.float64)
    if d.get("notch"):
        x = mne.filter.notch_filter(x, fs_in, freqs=[d["notch"]], verbose="ERROR")
    x = mne.filter.filter_data(x, fs_in, d["band"][0], d["band"][1], verbose="ERROR")
    fs = d["fs"]
    if int(fs_in) != int(fs):
        g = np.gcd(int(fs_in), int(fs))
        x = resample_poly(x, int(fs) // g, int(fs_in) // g, axis=1)
    return x.astype(np.float32)


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    """Standardise every channel over the recording file (Eq. 2)."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)
