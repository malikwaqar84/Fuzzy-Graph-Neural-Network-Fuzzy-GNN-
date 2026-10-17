"""Parsing of the CHB-MIT summary files, seizure bookkeeping and patient selection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

# chb21 was recorded from the same subject as chb01 (1.5 years later). For a
# patient-independent evaluation the two cases must never be split between the
# training and test sets, so they share one subject id.
SUBJECT_OF = {"chb21": "chb01"}


@dataclass
class EdfFile:
    name: str
    start_s: float | None          # absolute start time (s) from the first file of the case
    end_s: float | None
    seizures: list = field(default_factory=list)   # [(onset_s, offset_s)] relative to file start


@dataclass
class Case:
    case_id: str
    files: list

    @property
    def subject(self) -> str:
        return SUBJECT_OF.get(self.case_id, self.case_id)

    @property
    def has_times(self) -> bool:
        return all(f.start_s is not None for f in self.files)

    def seizures_abs(self):
        """All seizures of the case as (onset, offset) in absolute seconds, sorted."""
        out = []
        for f in self.files:
            for on, off in f.seizures:
                out.append((f.start_s + on, f.start_s + off))
        return sorted(out)

    def to_dict(self):
        return {"case_id": self.case_id, "files": [asdict(f) for f in self.files]}

    @staticmethod
    def from_dict(d):
        return Case(d["case_id"], [EdfFile(**f) for f in d["files"]])


def _clock_to_s(txt: str) -> float:
    h, m, s = [int(float(v)) for v in txt.strip().split(":")]
    return h * 3600 + m * 60 + s   # hours may be >= 24 in some summaries


def parse_summary(path: str | Path) -> Case:
    """Parse one chbXX-summary.txt file into a Case with absolute file times."""
    path = Path(path)
    case_id = path.name.split("-")[0]
    text = path.read_text(errors="ignore")
    files, cur = [], None
    pending_on = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("File Name:"):
            cur = EdfFile(name=line.split(":", 1)[1].strip(), start_s=None, end_s=None)
            files.append(cur)
        elif cur is None:
            continue
        elif line.startswith("File Start Time:"):
            cur.start_s = _clock_to_s(line.split(":", 1)[1])
        elif line.startswith("File End Time:"):
            cur.end_s = _clock_to_s(line.split(":", 1)[1])
        elif re.match(r"Seizure( \d+)? Start Time:", line):
            pending_on = float(re.findall(r"(\d+)\s*seconds", line)[0])
        elif re.match(r"Seizure( \d+)? End Time:", line) and pending_on is not None:
            cur.seizures.append((pending_on, float(re.findall(r"(\d+)\s*seconds", line)[0])))
            pending_on = None

    # Convert clock times to a continuous timeline (recordings can pass midnight).
    if files and all(f.start_s is not None and f.end_s is not None for f in files):
        day, prev_end = 0.0, None
        for f in files:
            s, e = f.start_s + day, f.end_s + day
            if prev_end is not None and s < prev_end - 60:
                day += 86400.0
                s += 86400.0
                e += 86400.0
            if e < s:
                e += 86400.0
            f.start_s, f.end_s = s, e
            prev_end = e
        t0 = files[0].start_s
        for f in files:
            f.start_s -= t0
            f.end_s -= t0
    else:
        for f in files:
            f.start_s = f.end_s = None
    return Case(case_id, files)


def lead_seizures(case: Case, lead_gap_s: float):
    """Onsets of lead seizures: seizures starting at least lead_gap_s after the end of the previous one."""
    lead, prev_off = [], None
    for on, off in case.seizures_abs():
        if prev_off is None or on - prev_off >= lead_gap_s:
            lead.append(on)
        prev_off = off if prev_off is None else max(prev_off, off)
    return lead
