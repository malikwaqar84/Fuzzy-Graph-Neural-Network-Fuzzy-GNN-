"""Download the CHB-MIT Scalp EEG Database (v1.0.0) from PhysioNet, with resume.

    python scripts/download_chbmit.py --out data/chbmit
    python scripts/download_chbmit.py --out data/chbmit --cases chb01 chb02
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

BASE = "https://physionet.org/files/chbmit/1.0.0/"


def fetch(session, url, dst: Path, retries=5):
    for attempt in range(retries):
        try:
            head = session.head(url, allow_redirects=True, timeout=60)
            size = int(head.headers.get("Content-Length", 0))
            if dst.exists() and size and dst.stat().st_size == size:
                return "exists"
            tmp = dst.with_suffix(dst.suffix + ".part")
            pos = tmp.stat().st_size if tmp.exists() else 0
            hdr = {"Range": f"bytes={pos}-"} if pos else {}
            with session.get(url, headers=hdr, stream=True, timeout=120) as r:
                if r.status_code == 200:
                    pos = 0                                  # server ignored the range request
                r.raise_for_status()
                with open(tmp, "ab" if pos else "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if size and tmp.stat().st_size != size:
                raise IOError(f"size mismatch for {url}")
            tmp.replace(dst)
            return "ok"
        except Exception as e:                               # noqa: BLE001
            if attempt == retries - 1:
                return f"failed: {e}"
    return "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chbmit")
    ap.add_argument("--cases", nargs="*", help="only these cases, e.g. chb01 chb02")
    args = ap.parse_args()
    out = Path(args.out)
    s = requests.Session()
    records = s.get(BASE + "RECORDS", timeout=60).text.split()
    cases = sorted({r.split("/")[0] for r in records})
    if args.cases:
        cases = [c for c in cases if c in args.cases]
    todo = [f"{c}/{c}-summary.txt" for c in cases] + [r for r in records if r.split("/")[0] in cases]
    todo += [r + ".seizures" for r in s.get(BASE + "RECORDS-WITH-SEIZURES", timeout=60).text.split()
             if r.split("/")[0] in cases]
    failed = []
    for rel in tqdm(todo, desc="CHB-MIT"):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        st = fetch(s, BASE + rel, dst)
        if st.startswith("failed"):
            failed.append((rel, st))
    print(f"{len(todo) - len(failed)} of {len(todo)} files present in {out}")
    for rel, st in failed:
        print("  ", rel, st)


if __name__ == "__main__":
    main()
