______________________________________________________________________

<div align="center">

# A fast and lightweight model for Causal Audio-Visual Speech Separation

<p align="center">
  <strong>Wendi Sang<sup>1,*</sup>, Kai Li<sup>2,*</sup>, Runxuan Yang<sup>2</sup>, Jianqiang Huang<sup>1</sup>, Xiaolin Hu<sup>2</sup></strong><br>
    <strong><sup>1</sup>Qinghai University, Xining, China</strong><br>
    <strong><sup>2</sup>Tsinghua University, Beijing, China</strong><br>
  <a href="https://arxiv.org/abs/2506.06689">ArXiv</a> | <a href="https://cslikai.cn/Swift-Net/">Demo</a>

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=JusperLee.Swift-Net" alt="访客统计" />
  <img src="https://img.shields.io/github/stars/JusperLee/Swift-Net?style=social" alt="GitHub stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey">
</p>

</div>

# Introduction

Here is the code for our proposed Swift-Net and some causal variants of the AVSS method. 

Ours: Swift-Net

Variants of the AVSS: av_convtasnet, av_dprnn, av_sepformer, av_tfgridenet, ctcnet, rtfsnet.


# Installation

Before you begin, ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system. Then, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/spkgyk/SwiftNet.git
   cd SwiftNet
   ```

2. Create a Conda environment with the required dependencies using the provided script:
   ```bash
   conda env create -f environment.yaml
   ```

**Note:** AVSS is a GPU-intensive task, so make sure you have access to a GPU for both installation and training.

# Datasets & Pretrained Video Model

1. Please download the pretrained video module used in our experiments by visiting [CTCNet](https://github.com/JusperLee/CTCNet)'s official repository. Instructions for downloading the various datasets used in our paper are also there, along with some additional scripts and utils for running the models in various real-world situations. 

2. Once the datasets are downloaded and formatted, please run the relevent scripts in the `data-preprocess` folder, i.e.
   ```bash
   python process_lrs23.py --in_audio_dir audio/wav16k/min --in_mouth_dir mouths --out_dir LRS2
   python process_lrs23.py --in_audio_dir audio/wav16k/min --in_mouth_dir mouths --out_dir LRS3
   python process_vox2.py --in_audio_dir audio/wav16k/min --in_mouth_dir mouths --out_dir VOX2
   ```

# Training & Evaluation

Training the AVSS model is a straightforward process using the `audio_train.py` script. You can customize the training by specifying a configuration file and, if necessary, a checkpoint for resuming training. Here's how to get started:

1. Run the training script with a default configuration file:
   ```bash
   python train.py --conf-dir config/lrs2_SwiftNet.yml
   ```

2. If you encounter unexpected interruptions during training and wish to resume from a checkpoint, use the following command (replace the checkpoint path with your specific checkpoint):
   ```bash
   python train.py --conf-dir config/lrs2_SwiftNet.yml \
   --checkpoint ../experiments/audio-visual/SwiftNet.yml/LRS2/4_layers/checkpoints/epoch=220-val_loss=-13.34.ckpt
   ```

Feel free to explore and fine-tune the configuration files in the `config` directory to suit your specific requirements and dataset.

Use the `audio_visual_test.py` script for evaluating your trained model on the test set. 

## Visual embedding cache, frequency mixer and testing

Three independent choices are available. Each one is controlled by the YAML config and works without the others.

| Choice | Config key | Values |
|---|---|---|
| Visual features | `datamodule.data_config.visual_features_path` | `null` (online FRCNN backbone) or a cache directory |
| Frequency modeling | `audionet.audionet_config.audio_params.frequency_mixer.type` | `grnn` (original, also the default when absent) or `conv` |
| FTGS depth/sharing | `audio_params.repeats`, `audio_params.shared` | any value |

**1. Precompute visual embeddings once** (deterministic center-crop preprocessing, FP32 by default):
```bash
python scripts/precompute_visual_embeddings.py \
    --conf_dir configs/lrs2_SwiftNet_6.yml \
    --output_dir DataPreProcess/LRS2_visual_features \
    --splits tr cv tt --batch_size 64 --num_workers 8
# re-run the same command after an interruption: completed entries are skipped
python scripts/precompute_visual_embeddings.py --conf_dir configs/lrs2_SwiftNet_6.yml \
    --output_dir DataPreProcess/LRS2_visual_features --verify_only
```
Add `--cpu` to run on CPU and `--dtype float16` to halve the storage.

**2. Check the numerical equivalence** of cached and online embeddings and of the separator output:
```bash
python scripts/validate_visual_cache.py --conf_dir configs/lrs2_SwiftNet_6.yml \
    --cache_dir DataPreProcess/LRS2_visual_features --split cv --num_samples 64 \
    --ckpt Experiments/LRS2_SwiftNet_6/checkpoints/last.ckpt --device cuda
```

**3. Train** (`configs/lrs2_SwiftNet_6_cached.yml` is the original SwiftNet-6 with cached embeddings):
```bash
python audio_train.py --conf_dir configs/lrs2_SwiftNet_6_cached.yml
python audio_train.py --conf_dir configs/lrs2_SwiftNet_6_freqmix_sum.yml     # conv mixer, sum fusion
python audio_train.py --conf_dir configs/lrs2_SwiftNet_6_freqmix_concat.yml  # conv mixer, concat fusion
```
Resume with `audio_train_resume.py --conf_dir <config> --ckpt_path <checkpoint>`. Initialize from another run with
`audio_train.py ... --init_ckpt <checkpoint> --init_mode strict|partial` (`partial` loads only matching tensors and reports the rest).

**4. Test** (the only place where the test split is evaluated; training only validates):
```bash
python scripts/test_lightning_checkpoint.py --conf_dir configs/lrs2_SwiftNet_6_cached.yml \
    --ckpt_path Experiments/<exp>/checkpoints/last.ckpt
```

**Validation metric.** The monitored metric is `val_loss` (it was `val_loss/dataloader_idx_0`). Resuming an older checkpoint
renames its saved callback state automatically into a copy in the new experiment directory. The original file is not modified.

**Tests and benchmarks**
```bash
python -m unittest tests.test_frequency_mixer tests.test_pipeline -v
python scripts/benchmark_frequency_mixer.py --device cuda --full_model --output results/freq_bench.json
python scripts/benchmark_training_pipeline.py --device cuda --precision fp32 --output results/train_bench.json
```

# Contact

If you have any questions, suggestions, or encounter any issues, please don't hesitate to reach out to us. You can contact us via email at `tsinghua.kaili [AT] gmail.com`.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Citations

If you find this code useful in your research, please cite our work:
```
@misc{sang2025fastlightweightmodelcausal,
      title={A Fast and Lightweight Model for Causal Audio-Visual Speech Separation}, 
      author={Wendi Sang and Kai Li and Runxuan Yang and Jianqiang Huang and Xiaolin Hu},
      year={2025},
      eprint={2506.06689},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2506.06689}, 
}
```