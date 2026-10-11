"""Disk cache of precomputed visual embeddings (one entry per unique visual source).

Layout of a cache directory:

    metadata.json   identity of the backbone, preprocessing and storage format
    index.json      {"keys": [...]}; row i of the arrays below belongs to keys[i]
    features.npy    memmap, shape [N, D, T], dtype float32 or float16, zero padded after lengths[i]
    lengths.npy     uint16 [N], number of valid frames of each entry
    status.npy      uint8 [N], 1 once the row was fully written and flushed

Why a contiguous memmap:
    Entries have a fixed size (D * T * itemsize, e.g. 512 * 50 * 4 = 100 KiB). A single memmap gives
    O(1) random access without per-sample file opens, no serialization, no compression, and it is
    safe for several DataLoader workers because each process opens its own mapping lazily.
    A dataset of ~19k visual sources is about 2 GB in FP32, so it could also be loaded in RAM.
    Writes are chunked and status is flushed only after features are flushed, so an interrupted
    run can resume: rows with status 0 are recomputed, rows with status 1 are never touched.

Keys are normalized paths of the mouth ``.npz`` files, i.e. the identity of the visual source,
not of the audio mixture (one visual utterance can appear in several mixtures).
"""

import json
import os

import numpy as np
import torch

CACHE_FORMAT_VERSION = 1

METADATA_FILE = "metadata.json"
INDEX_FILE = "index.json"
FEATURES_FILE = "features.npy"
LENGTHS_FILE = "lengths.npy"
STATUS_FILE = "status.npy"

# Metadata fields that must match exactly when a cache is reused or read.
IDENTITY_FIELDS = (
    "cache_format_version",
    "backbone",
    "preprocessing",
    "input_fps",
    "segment_duration",
    "num_frames",
    "feature_dim",
    "feature_dtype",
)


def normalize_visual_key(path):
    """Canonical identity of a visual source (mouth ``.npz`` path)."""
    return os.path.normpath(str(path))


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def create_cache(root, keys, metadata):
    """Create an empty cache for ``keys``. Refuses to overwrite an existing cache."""
    if os.path.exists(os.path.join(root, METADATA_FILE)):
        raise FileExistsError(f"A visual cache already exists in {root}; resume it instead of recreating it.")
    os.makedirs(root, exist_ok=True)

    n = len(keys)
    keys = [normalize_visual_key(k) for k in keys]
    if len(set(keys)) != n:
        raise ValueError("Duplicate visual keys passed to create_cache")

    meta = dict(metadata)
    meta["num_entries"] = n
    meta["layout"] = "features.npy [N, feature_dim, num_frames]; zero padded after lengths[i]"

    np.lib.format.open_memmap(
        os.path.join(root, FEATURES_FILE),
        mode="w+",
        dtype=np.dtype(meta["feature_dtype"]),
        shape=(n, meta["feature_dim"], meta["num_frames"]),
    ).flush()
    np.save(os.path.join(root, LENGTHS_FILE), np.zeros(n, dtype=np.uint16))
    np.save(os.path.join(root, STATUS_FILE), np.zeros(n, dtype=np.uint8))
    _write_json(os.path.join(root, INDEX_FILE), {"keys": keys})
    _write_json(os.path.join(root, METADATA_FILE), meta)


def check_metadata(expected, found):
    """Raise if the cache on disk was built with a different backbone or preprocessing."""
    mismatches = []
    for field in IDENTITY_FIELDS:
        if expected.get(field) != found.get(field):
            mismatches.append(f"{field}: cache={found.get(field)!r} expected={expected.get(field)!r}")
    if mismatches:
        raise ValueError("Incompatible visual cache:\n  " + "\n  ".join(mismatches))


class VisualFeatureCache:
    """Read access to a cache directory. Safe to pickle to DataLoader workers (memmaps are opened lazily)."""

    def __init__(self, root):
        self.root = root
        self.metadata = _read_json(os.path.join(root, METADATA_FILE))
        if self.metadata.get("cache_format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported visual cache format {self.metadata.get('cache_format_version')} in {root}"
            )
        self.keys = _read_json(os.path.join(root, INDEX_FILE))["keys"]
        self.row = {k: i for i, k in enumerate(self.keys)}
        self.status = np.load(os.path.join(root, STATUS_FILE))
        self.lengths = np.load(os.path.join(root, LENGTHS_FILE))
        self.feature_dim = self.metadata["feature_dim"]
        self.num_frames = self.metadata["num_frames"]
        self._features = None
        self._pid = None

    def __len__(self):
        return len(self.keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None  # never pickle the memmap (it would copy the whole file)
        state["_pid"] = None
        return state

    @property
    def features(self):
        # Re-open after fork, so each worker has its own mapping.
        if self._features is None or self._pid != os.getpid():
            self._features = np.load(os.path.join(self.root, FEATURES_FILE), mmap_mode="r")
            self._pid = os.getpid()
        return self._features

    def __contains__(self, key):
        return normalize_visual_key(key) in self.row

    def get(self, key):
        """Return the embedding of ``key`` as a float32 tensor ``[feature_dim, length]``."""
        key = normalize_visual_key(key)
        if key not in self.row:
            raise KeyError(f"Visual source not in cache {self.root}: {key}")
        i = self.row[key]
        if self.status[i] != 1:
            raise RuntimeError(f"Visual cache entry not completed (run precompute again): {key}")
        length = int(self.lengths[i])
        return torch.from_numpy(np.array(self.features[i, :, :length], dtype=np.float32))

    def missing_keys(self, keys):
        """Keys that are absent from the index or not completed. Used to fail fast before training."""
        bad = []
        for k in keys:
            k = normalize_visual_key(k)
            i = self.row.get(k)
            if i is None or self.status[i] != 1:
                bad.append(k)
        return bad


def write_rows(root, rows, embeddings, lengths):
    """Write a chunk of finished entries. ``embeddings`` is ``[n, D, T]``, ``lengths`` is ``[n]``.

    Features are flushed before the status bitmap, so a crash never marks an unwritten row as done.
    """
    features = np.load(os.path.join(root, FEATURES_FILE), mmap_mode="r+")
    features[rows] = embeddings
    features.flush()
    del features

    lens = np.load(os.path.join(root, LENGTHS_FILE))
    lens[rows] = lengths
    np.save(os.path.join(root, LENGTHS_FILE), lens)

    status = np.load(os.path.join(root, STATUS_FILE))
    status[rows] = 1
    np.save(os.path.join(root, STATUS_FILE), status)


def verify_cache(root, chunk=1024):
    """Check that every completed entry is finite and has a valid length. Returns a summary dict."""
    cache = VisualFeatureCache(root)
    feats = np.load(os.path.join(root, FEATURES_FILE), mmap_mode="r")
    done = np.flatnonzero(cache.status == 1)
    bad_nonfinite, bad_length = [], []
    for start in range(0, len(done), chunk):
        rows = done[start : start + chunk]
        finite = np.isfinite(feats[rows]).all(axis=(1, 2))
        bad_nonfinite.extend(rows[~finite].tolist())
        lens = cache.lengths[rows]
        bad_length.extend(rows[(lens < 1) | (lens > cache.num_frames)].tolist())
    return {
        "num_entries": len(cache),
        "completed": int(len(done)),
        "pending": int(len(cache) - len(done)),
        "non_finite_rows": bad_nonfinite,
        "invalid_length_rows": bad_length,
    }
