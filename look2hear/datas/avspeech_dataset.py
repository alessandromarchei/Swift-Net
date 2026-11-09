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
import json
from typing import Dict, Iterable, List, Iterator
from .transform import get_preprocessing_pipelines
from .visual_cache import VisualFeatureCache, normalize_visual_key


SOURCE_VISUAL_FPS = 25.0  # native frame rate of the mouth crops; lower rates are picked by frame index


def validate_visual_fps(visual_fps):
    """Return visual_fps as float, checking it is a usable rate for the mouth crops (0, 25]."""
    fps = float(visual_fps)
    if not (0.0 < fps <= SOURCE_VISUAL_FPS):
        raise ValueError(
            f"visual_fps must be in (0, {SOURCE_VISUAL_FPS:g}], got {visual_fps}. "
            "Mouth crops are 25 fps; rates above it cannot be recovered."
        )
    return fps


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
        visual_fps: float = SOURCE_VISUAL_FPS,
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
        # Frame rate of the mouth sequences this run consumes (the .npz files must already be at this rate).
        self.visual_fps = validate_visual_fps(visual_fps)
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            # Frames covering the segment at this run's visual rate (25 fps -> int(segment * 25), as before).
            self.fps_len = int(round(segment * self.visual_fps))
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

            mix_source, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            source = sf.read(
                self.sources[idx][0], start=rand_start, stop=stop, dtype="float32"
            )[0]
            if self.visual_cache is not None:
                source_mouth = self.visual_cache.get(self.sources[idx][1])  # [D, T] embedding
            else:
                source_mouth = self.lipreading_preprocessing_func(
                    np.load(self.sources[idx][1])["data"]
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

            mix_source, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            sources = []
            for src in self.sources[idx]:
                # import pdb; pdb.set_trace()
                sources.append(
                    sf.read(src[0], start=rand_start, stop=stop, dtype="float32")[0]
                )
            # import pdb; pdb.set_trace()
            if self.visual_cache is not None:
                sources_mouths = torch.stack([self.visual_cache.get(src[1]) for src in self.sources[idx]])
            else:
                sources_mouths = torch.stack(
                    [
                        torch.from_numpy(
                            self.lipreading_preprocessing_func(np.load(src[1])["data"])
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
        visual_fps: float = SOURCE_VISUAL_FPS,
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        # None: online mouth frames + FRCNNVideoModel. Path: precomputed embeddings (see scripts/precompute_visual_embeddings.py).
        self.visual_features_path = visual_features_path
        # Visual frame rate of the run (<= 25). Must match the mouth .npz files and any precomputed cache.
        self.visual_fps = validate_visual_fps(visual_fps)

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
        if cache is not None:
            # A cache built at another rate would silently misalign video and audio.
            cache_fps = float(cache.metadata["input_fps"])
            if cache_fps != self.visual_fps:
                raise RuntimeError(
                    f"Visual cache {self.visual_features_path} was built at {cache_fps:g} fps, "
                    f"but this run uses visual_fps={self.visual_fps:g}. "
                    "Rebuild it with scripts/precompute_visual_embeddings.py --visual_fps "
                    f"{self.visual_fps:g}."
                )
        self.data_train = AVSpeechDataset(
            json_dir = self.train_dir,
            n_src = self.n_src,
            sample_rate = self.sample_rate,
            segment = self.segment,
            normalize_audio = self.normalize_audio,
            is_train=True,
            visual_cache=cache,
            visual_fps=self.visual_fps,
        )
        self.data_val = AVSpeechDataset(
            json_dir = self.valid_dir,
            n_src = self.n_src,
            sample_rate = self.sample_rate,
            segment = self.segment,
            normalize_audio = self.normalize_audio,
            is_train=False,
            visual_cache=cache,
            visual_fps=self.visual_fps,
        )
        self.data_test = AVSpeechDataset(
            json_dir = self.test_dir,
            n_src = self.n_src,
            sample_rate = self.sample_rate,
            segment = self.segment,
            normalize_audio = self.normalize_audio,
            is_train=False,
            visual_cache=cache,
            visual_fps=self.visual_fps,
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
