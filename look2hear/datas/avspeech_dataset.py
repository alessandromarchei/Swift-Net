import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile
import io
import json
from typing import Dict, Iterable, List, Iterator

import lmdb

from .transform import get_preprocessing_pipelines
from .visual_cache import VisualFeatureCache, normalize_visual_key


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class AVSpeechDataset(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        is_train: bool = True,
        visual_cache: VisualFeatureCache = None,
        lmdb_path: str = None,
    ):
        super().__init__()
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        # If set, the dataset returns cached embeddings [D, T] instead of mouth crops.
        self.visual_cache = visual_cache
        # If set, audio and mouths are read from this LMDB instead of the paths in the JSON files.
        self.lmdb_path = str(lmdb_path) if lmdb_path else None
        self._lmdb_env = None
        if self.lmdb_path is not None and not os.path.isdir(self.lmdb_path):
            raise FileNotFoundError(f"LMDB directory not found: {self.lmdb_path}")
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            self.fps_len = int(segment * 25)
        self.n_src = n_src
        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        # JSON entries written for LMDB use virtual "/lmdb/<split>/..." paths; they cannot be read from disk.
        if self.lmdb_path is None and any(
            str(entry[0]).startswith("/lmdb/") for entry in mix_infos
        ):
            raise ValueError(
                f"{json_dir} indexes LMDB entries (/lmdb/...), but no LMDB path was given. "
                "Set train_lmdb / valid_lmdb / test_lmdb in the datamodule config."
            )

        self.mix = []
        self.sources = []
        if self.n_src == 1:
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            print(len(mix_infos), len(sources_infos[0]), len(sources_infos[1]))
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

        elif self.n_src == 2:
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        self.mix.append(mix_infos[i])
                        self.sources.append([src_inf[i] for src_inf in sources_infos])

            else:
                for i in range(len(mix_infos)):
                    self.mix.append(mix_infos[i])
                    self.sources.append([sources_infos[0][i], sources_infos[1][i]])
            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

    def __len__(self):
        return self.length

    def _lmdb_get(self, key: str) -> bytes:
        if self._lmdb_env is None:
            # Opened lazily so each DataLoader worker creates its own environment.
            self._lmdb_env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
                subdir=True,
            )
        with self._lmdb_env.begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))
        if value is None:
            raise KeyError(f"LMDB key not found: {key} (database={self.lmdb_path})")
        return value

    def _read_audio(self, path, start=0, stop=None):
        """Read a wav listed in the JSON index, from LMDB (audio/<kind>/<name>) or from disk."""
        if self.lmdb_path is None:
            return sf.read(path, start=start, stop=stop, dtype="float32")[0]
        kind = os.path.basename(os.path.dirname(path))
        key = f"audio/{kind}/{os.path.basename(path)}"
        audio, _ = sf.read(io.BytesIO(self._lmdb_get(key)), start=start, stop=stop, dtype="float32")
        return audio

    def _read_mouth(self, path):
        """Raw mouth array (the 'data' field of the .npz), from LMDB (mouth/<name>.npz) or from disk."""
        if self.lmdb_path is None:
            return np.load(path)["data"]
        key = f"mouth/{os.path.basename(path)}"
        with np.load(io.BytesIO(self._lmdb_get(key))) as npz:
            return npz["data"]

    def visual_keys(self):
        """Identity (mouth .npz path) of every visual source used by this dataset."""
        keys = []
        for src in self.sources:
            if self.n_src == 1:
                keys.append(normalize_visual_key(src[1]))
            else:
                keys.extend(normalize_visual_key(s[1]) for s in src)
        return sorted(set(keys))

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        if self.n_src == 1:
            rand_start = 0
            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len

            mix_source = self._read_audio(self.mix[idx][0], rand_start, stop)
            source = self._read_audio(self.sources[idx][0], rand_start, stop)
            if self.visual_cache is not None:
                source_mouth = self.visual_cache.get(self.sources[idx][1])  # [D, T] embedding
            else:
                source_mouth = self.lipreading_preprocessing_func(
                    self._read_mouth(self.sources[idx][1])
                )[:, : self.fps_len]

            source = torch.from_numpy(source)
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)
            return mixture, source, source_mouth, self.mix[idx][0].split("/")[-1]

        if self.n_src == 2:
            rand_start = 0
            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len

            mix_source = self._read_audio(self.mix[idx][0], rand_start, stop)
            sources = []
            for src in self.sources[idx]:
                # import pdb; pdb.set_trace()
                sources.append(self._read_audio(src[0], rand_start, stop))
            # import pdb; pdb.set_trace()
            if self.visual_cache is not None:
                sources_mouths = torch.stack([self.visual_cache.get(src[1]) for src in self.sources[idx]])
            else:
                sources_mouths = torch.stack(
                    [
                        torch.from_numpy(
                            self.lipreading_preprocessing_func(self._read_mouth(src[1]))
                        )
                        for src in self.sources[idx]
                    ]
                )[:, : self.fps_len]
            # import pdb; pdb.set_trace()
            sources = torch.stack([torch.from_numpy(source) for source in sources])
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

            return mixture, sources, sources_mouths, self.mix[idx][0].split("/")[-1]

class AVSpeechDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        visual_features_path: str = None,
        train_lmdb: str = None,
        valid_lmdb: str = None,
        test_lmdb: str = None,
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        # None: online mouth frames + FRCNNVideoModel. Path: precomputed embeddings (see scripts/precompute_visual_embeddings.py).
        self.visual_features_path = visual_features_path
        # Per-split LMDB databases (tr.lmdb / cv.lmdb / tt.lmdb). None: read the files named in the JSON.
        self.train_lmdb = train_lmdb
        self.valid_lmdb = valid_lmdb
        self.test_lmdb = test_lmdb
        if self.visual_features_path is not None and any(
            [train_lmdb, valid_lmdb, test_lmdb]
        ):
            raise ValueError(
                "visual_features_path (precomputed embeddings) reads mouth files from disk and "
                "cannot be combined with the LMDB options yet."
            )

        # this line allows to access init params with 'self.hparams' attribute
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    @property
    def uses_cached_visual(self):
        return self.visual_features_path is not None

    def setup(self) -> None:
        cache = VisualFeatureCache(self.visual_features_path) if self.uses_cached_visual else None
        self.data_train = AVSpeechDataset(
            json_dir = self.train_dir,
            n_src = self.n_src,
            sample_rate = self.sample_rate,
            segment = self.segment,
            normalize_audio = self.normalize_audio,
            is_train=True,
            visual_cache=cache,
            lmdb_path=self.train_lmdb,
        )
        self.data_val = AVSpeechDataset(
            json_dir = self.valid_dir,
            n_src = self.n_src,
            sample_rate = self.sample_rate,
            segment = self.segment,
            normalize_audio = self.normalize_audio,
            is_train=False,
            visual_cache=cache,
            lmdb_path=self.valid_lmdb,
        )
        self.data_test = AVSpeechDataset(
            json_dir = self.test_dir,
            n_src = self.n_src,
            sample_rate = self.sample_rate,
            segment = self.segment,
            normalize_audio = self.normalize_audio,
            is_train=False,
            visual_cache=cache,
            lmdb_path=self.test_lmdb,
        )
        if cache is not None:
            # Fail before training if the cache does not cover every visual source of every split.
            for name, dataset in [("train", self.data_train), ("val", self.data_val), ("test", self.data_test)]:
                missing = cache.missing_keys(dataset.visual_keys())
                if missing:
                    raise RuntimeError(
                        f"Visual cache {self.visual_features_path} misses {len(missing)} entries "
                        f"of the {name} split (first: {missing[0]}). Run scripts/precompute_visual_embeddings.py."
                    )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test

# if __name__ == "__main__":
#     from tqdm import tqdm
#     val_set = AVSpeechDataset(
#         "/home/likai/nichang-stream/code/local/LRS2/test/",
#         n_src=2,
#         sample_rate=16000,
#         # segment=conf["data"]["segment"],
#         segment=3.0,
#         normalize_audio=False
#     )
#     for idx in tqdm(range(len(val_set))):
#         mixture, sources, sources_mouths, _ = val_set[idx]
#         import pdb; pdb.set_trace()
