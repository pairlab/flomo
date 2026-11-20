# FloMo

**Any-to-any forward-dynamics cotraining for robot learning from video.**

FloMo fine-tunes the **Wan 2.2 TI2V-5B** video model (LoRA by default) to predict any
subset of `{robot action, future video, motion tokens}` from `{image, language}`.
Action is only one output, so one model learns from robot data and from action-free
human video. The paper trains on EgoVerse (egocentric human video), MolmoAct2, and
RealYAM (real robot), then fine-tunes on RoboTwin 2.0 (simulation).

**Contents:** [Install](#install) · [Pretrained checkpoints](#pretrained-checkpoints) ·
[Inference](#inference) · [Fine-tune on RoboTwin 2.0](#fine-tune-on-robotwin-20) ·
[Train on your own data](#train-on-your-own-data) · [Preprocessing](#preprocessing) ·
[Motion tokens](#motion-tokens) · [Code layout](#code-layout)

## Install

Needs a CUDA GPU and Python 3.10. Run all commands from the repo root.

```bash
conda create -n flomo python=3.10 -y && conda activate flomo
git clone https://github.com/pairlab/flomo.git && cd flomo

# Backbone code, imported from the repo root
git clone https://github.com/Wan-Video/Wan2.2.git
git clone https://github.com/dreamzero0/dreamzero.git

# PyTorch (cu128 shown; Blackwell needs CUDA 12.8+), then FloMo
pip install "torch==2.10.*" "torchvision==0.25.*" --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# Wan VAE and text-encoder weights
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir ./wan_checkpoints
```

The Wan DiT and the umT5 tokenizer download on first use.

To preprocess your own videos, also install the 3D trackers in the repo root:

```bash
git clone https://github.com/henry123-boy/SpaTrackerV2.git
git clone https://github.com/facebookresearch/vggt.git
pip install "numpy<2" kornia==0.8.1 moviepy==1.0.0 pycolmap==3.11.1 pyceres==2.4 \
    scikit-learn flow_vis jaxtyping \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@d3a577acf0a9ad7e513a1416449a07b6f47d967f"
```

- Keep `torchvision<=0.25`: SpaTrackerV2 uses `torchvision.io.write_video`, removed in 0.26.
- Keep `numpy<2` in the same command, or pip installs numpy 2.
- Skip SpaTrackerV2's `requirements.txt`: its `xformers` pin needs an older torch.

## Pretrained checkpoints

The paper base models are at [shitijgovil/FloMo](https://huggingface.co/shitijgovil/FloMo).
They share training data and differ only in outputs:

| File | Predicts |
|---|---|
| `ma/flomo_ma.pt` | motion tokens + action |
| `va/flomo_va.pt` | video + action |
| `vma/flomo_vma.pt` | video + motion tokens + action |

```bash
hf download shitijgovil/FloMo vma/flomo_vma.pt --local-dir ./flomo_ckpts
```

MA and VMA include the per-dataset motion-token stats that `decode_positions` needs.
Copies downloaded before 2026-11-19 lack them; download again.

## Inference

`infer_example.py` maps camera images, a task, and the current end-effector pose to a
`(T, 14)` action chunk: `x, y, z, yaw, pitch, roll, gripper` per arm.

```bash
python infer_example.py --checkpoint ./flomo_ckpts/vma/flomo_vma.pt \
    --wan-ckpt-dir ./wan_checkpoints \
    --image primary.png --image left.png --image right.png \
    --task "pick up the croissant and place it in the oven" \
    --ee-pose="x,y,z,yaw,pitch,roll,grip,x,y,z,yaw,pitch,roll,grip"
```

- One `--image` per camera, in order `primary`, `left`, `right`.
- Use `--ee-pose=...` with `=`; otherwise a negative first value reads as a flag.
- `--normalizer-type`: `realyam` (default) for base checkpoints, `robotwin` for RoboTwin fine-tunes.

The script only predicts actions; camera capture and robot control are up to you.

## Fine-tune on RoboTwin 2.0

Fine-tunes one task (`open_laptop`) from a base checkpoint, as in the paper: 50
Aloha-AgileX `demo_clean` demos, 5,000 updates at batch size 256. Needs the trackers
from [Install](#install).

```bash
# 1. Download demos
hf download TianxingChen/RoboTwin2.0 dataset/open_laptop/demo_clean.zip \
    --repo-type dataset --local-dir <RoboTwin>/data/hf
unzip <RoboTwin>/data/hf/dataset/open_laptop/demo_clean.zip -d <RoboTwin>/data/demo_clean

# 2. Preprocess: 3D tracks and text embeddings
for mode in tracks text_emb; do
  python -m preprocessing.preprocess_robotwin mode=$mode task_names=[open_laptop] \
      source_dir=<RoboTwin>/data/demo_clean dest_dir=<RoboTwin>/data/flomo
done

# 3. Fine-tune (resume=false: weights only, fresh optimizer and step count)
python train.py --config-name train_robotwin data_root=<RoboTwin>/data \
    robotwin_task=open_laptop checkpoint=./flomo_ckpts/vma/flomo_vma.pt \
    resume=false run_name=robotwin_open_laptop
```

Checkpoints go to `checkpoints/forward_dynamics/robotwin_open_laptop/<epoch>.pt`. VA
and VMA warn about unused `video_primary` keys; this is expected, since the RoboTwin
config does not predict video. Run the result with [Inference](#inference) and
`--normalizer-type robotwin`; score it with RoboTwin's policy evaluator.

## Train on your own data

`cfg/datasets/` configs choose the data and each dataset's inputs and outputs.
`cfg/train.yaml` holds training settings; its comments explain every key.

| `datasets=` | Use |
|---|---|
| `template` (default) | Annotated example; copy it and fill in the `???` fields. |
| `flomo` | Paper mixture (EgoVerse + MolmoAct2 + RealYAM). |
| `robotwin` | One RoboTwin task (with `--config-name train_robotwin`). |

```bash
python train.py datasets=my_dataset run_name=my_run

# From a base checkpoint (weights only)
python train.py datasets=my_dataset checkpoint=./flomo_ckpts/vma/flomo_vma.pt resume=false run_name=my_ft

# Paper mixture, 8 GPUs
torchrun --nproc_per_node=8 train.py datasets=flomo data_root=/path/to/flomo_data run_name=flomo_repro
```

- **Resume:** with `resume=true` (default), a run with the same `run_name` continues
  from its last checkpoint.
- **Overrides:** set any key on the command line, e.g. `optim.lr=5e-5 batch_size=64
  gpu_max_bs=16`. `batch_size` is global; `gpu_max_bs` is the per-GPU micro-batch.
- **Cache:** the first run caches per-dataset motion stats in
  `~/.cache/flomo/track_stats` (override with `FLOMO_CACHE_DIR`).
- **`flomo.yaml` defaults:** the `dataset_defaults` anchor expands before command-line
  overrides, so `datasets.0.args.sample_stride=2` changes only the first dataset.

## Preprocessing

Each dataset has `preprocessing/preprocess_<dataset>.py` and
`cfg/preprocessing/preprocess_<dataset>.yaml`. `mode=` selects a stage:

| Dataset | Modes |
|---|---|
| RoboTwin | `tracks`, `text_emb`, `vae_latent_h16` |
| RealYAM | `tracks`, `text_emb`, `vae_latent_h16` |
| MolmoAct2 | `tracks`, `video`, `text`, `text_emb`, `vae_latent`, `vae_latent_h16`, `actions` |
| EgoVerse | `tracks`, `video`, `text`, `text_emb`, `meta`, `vae_latent_h16` |

- `tracks` makes the 3D tracks for motion tokens; `text_emb` caches umT5 embeddings.
- **EgoVerse:** run `video` before `tracks` and `vae_latent_h16`, and `text` before
  `text_emb`. `datasets=flomo` needs `video`, `tracks`, `vae_latent_h16`, `text`, and
  `text_emb`. Set training `data_root` to the preprocessing `dest_dir`; dataset names
  are `<output_name>_<split>`.
- **MolmoAct2 subsets:** build an episode index, select hours, then preprocess:

  ```bash
  python -m preprocessing.preprocess_molmoact2 index --index-dir /idx build --source-dir /raw
  python -m preprocessing.preprocess_molmoact2 index --index-dir /idx select --name subset --hours 10
  python -m preprocessing.preprocess_molmoact2 index_path=/idx/subset/index.jsonl mode=video
  ```

## Motion tokens

Motion tokens are dense 3D scene flow, encoded by the frozen Wan VAE as a 3-channel video:

1. **Track** (`preprocessing/SpatialTracker.py`): SpatialTrackerV2 tracks a 64×64 grid
   (4,096 points) in 3D, in the camera frame of each window's first frame, restarting
   every 16 frames.
2. **Load** (`BaseDataset.load_tracks`): with `supervision=cumulative_disp`, each
   16-frame window becomes displacement from frame 0: `(1, 16, 4096, 3)`.
3. **Encode** (`MotionTokensWanPretrainedModality`): an XYZ grid `(3, 15, 64, 64)` is
   resized to 256, scaled per dataset to `[-1, 1]` (`flomo/utils/motion_stats.py`),
   and VAE-encoded to a `(48, 5, 16, 16)` latent, which the Wan patch embedding tokenizes.

`supervision=point_tracks` and `optical_flow` encode per-frame changes instead. All
datasets that output motion tokens must use the same mode.

## Code layout

```
train.py                 # training entry point (Hydra)
infer_example.py         # single-step action inference example
cfg/
  train.yaml, train_robotwin.yaml
  datasets/              # template, flomo, robotwin
  preprocessing/         # one config per preprocess_<dataset>.py
flomo/
  models/FloMo.py        # Wan trunk, token order, flow-matching timesteps, _build_modality
  models/modalities.py   # Modality contract + image/video/text/action/motion-token modalities
  loaders/               # BaseDataset, per-dataset readers, cotrain_loader (sampling, batching)
  utils/                 # normalizers, motion stats, Wan encoding, training, visualization
preprocessing/           # trackers (SpatialTracker.py, VGGT.py), writers, per-dataset pipelines
```

- **New dataset:** subclass `BaseDataset` and register it in `cotrain_loader._dataset_class`.
- **New modality:** implement the `Modality` contract in `models/modalities.py` and add
  a `kind` branch to `_build_modality`.

## License

MIT. See [LICENSE](LICENSE).
