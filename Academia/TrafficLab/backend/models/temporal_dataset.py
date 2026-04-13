# temporal_dataset.py
import os
import re
import glob
import random
from typing import List, Optional, Tuple, Dict, Any, Callable

import torch
from torch.utils.data import Dataset
# --- Safe allowlist for PyG objects so torch.load(..., weights_only=True) works ---
from torch_geometric.data import Data, Batch
try:
    # PyG registers this alias/class internally; needed by the restricted unpickler
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
except Exception:
    DataEdgeAttr = None
    DataTensorAttr = None
    GlobalStorage = None

# Handle PyTorch version compatibility for add_safe_globals
try:
    from torch.serialization import add_safe_globals
    _safe_objs = [Data, Batch]
    if DataEdgeAttr is not None:
        _safe_objs.append(DataEdgeAttr)
    if DataTensorAttr is not None:
        _safe_objs.append(DataTensorAttr)
    if GlobalStorage is not None:
        _safe_objs.append(GlobalStorage)
    add_safe_globals(_safe_objs)
except ImportError:
    # add_safe_globals not available in older PyTorch versions
    pass
# -------------------------------------------------------------------------------



def extract_step_number(path: str) -> int:
    m = re.search(r"step_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


class TemporalGraphDataset(Dataset):
    """
    Builds sliding windows over sorted *.pt snapshots.
    Each item = dict with:
      - time_slices: List[Data] of length T (or shorter if allow_incomplete_tail)
      - file_paths:  List[str] matching time_slices
      - steps:       List[int] extracted from filenames

    Shuffling behavior:
      - If shuffle_windows=True, the *order of windows* is shuffled once at init (optionally with a seed).
      - The temporal order *inside* each window is always preserved.
    """
    def __init__(
        self,
        root: str,
        window_size: int,
        stride_size: int,
        num_files: Optional[int] = None,
        start_idx: int = 0,
        file_pattern: str = "*.pt",
        sort_key: Optional[Callable[[str], tuple]] = None,
        allow_incomplete_tail: bool = False,
        shuffle_windows: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert window_size >= 1, "window_size must be >= 1"
        assert stride_size >= 1, "stride_size must be >= 1"

        self.root = root
        self.window_size = window_size
        self.stride_size = stride_size
        self.start_idx = max(0, int(start_idx))
        self.allow_incomplete_tail = allow_incomplete_tail
        self.shuffle_windows = bool(shuffle_windows)
        self.seed = seed

        # Load and sort files
        files = sorted(
            glob.glob(os.path.join(root, file_pattern)),
            key=(sort_key if sort_key is not None else lambda p: (extract_step_number(p), os.path.basename(p)))
        )
        if num_files is not None:
            num_files = max(0, int(num_files))
            files = files[self.start_idx:self.start_idx + num_files]
        else:
            files = files[self.start_idx:]

        if len(files) == 0:
            raise FileNotFoundError(f"No snapshot .pt files found under {root} with pattern {file_pattern}")

        self.files: List[str] = files
        self.steps: List[int] = [extract_step_number(p) for p in self.files]

        # Build window index list (start,end_exclusive)
        self._windows: List[Tuple[int, int]] = []
        last_start = len(self.files) - (window_size if not allow_incomplete_tail else 1)
        if last_start < 0:
            if allow_incomplete_tail:
                self._windows.append((0, len(self.files)))
            else:
                raise ValueError(
                    f"Not enough files ({len(self.files)}) for window_size={window_size} with allow_incomplete_tail=False"
                )
        else:
            i = 0
            while i <= len(self.files) - window_size:
                self._windows.append((i, i + window_size))
                i += self.stride_size
            if allow_incomplete_tail and i < len(self.files):
                self._windows.append((i, len(self.files)))

        # Optional: shuffle the order of windows (never the order inside each window)
        if self.shuffle_windows and len(self._windows) > 1:
            rng = random.Random(self.seed)
            rng.shuffle(self._windows)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start, end = self._windows[idx]
        paths = self.files[start:end]
        datas: List[Data] = [torch.load(p, weights_only=False) for p in paths]
        return {
            "time_slices": datas,
            "file_paths": paths,
            "steps": self.steps[start:end],
        }


def temporal_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate into:
      - time_batches: List[Batch] of length T
      - steps, file_paths: per-sample metadata
      - meta: dict with B and T
    Trims to the minimum T if allow_incomplete_tail created shorter windows.
    """
    assert len(batch) > 0
    lengths = [len(s["time_slices"]) for s in batch]
    T = min(lengths)
    if T == 0:
        raise RuntimeError("Received an empty time window in collate.")

    time_batches: List[Batch] = []
    for t in range(T):
        time_batches.append(Batch.from_data_list([s["time_slices"][t] for s in batch]))

    return {
        "time_batches": time_batches,
        "steps": [s["steps"][:T] for s in batch],
        "file_paths": [s["file_paths"][:T] for s in batch],
        "meta": {"batch_size": len(batch), "T": T},
    }
