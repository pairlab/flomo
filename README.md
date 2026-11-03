# FloMo

**Any-to-any forward-dynamics cotraining for robot learning from video.**

FloMo fine-tunes the **Wan 2.2 TI2V-5B** video model (LoRA by default) to predict any
subset of `{robot action, future video, motion tokens}` from `{image, language}`.
Because action is only one of the outputs, one model can learn from robot data with
actions and from human video without actions.

The paper trains on EgoVerse (egocentric human video), MolmoAct2, and RealYAM (real
robot), then fine-tunes on RoboTwin 2.0 (simulation).

**Contents:** [Install](#install) · [Pretrained checkpoints](#pretrained-checkpoints) ·
[Inference](#inference) · [Fine-tune on RoboTwin 2.0](#fine-tune-on-robotwin-20) ·
[Train on your own data](#train-on-your-own-data) · [Preprocessing reference](#preprocessing-reference) ·
[Motion tokens](#motion-tokens) · [Code layout](#code-layout)

## Install

You need a CUDA GPU and Python 3.10. Run every command from the repo root.

```bash
conda create -n flomo python=3.10 -y && conda activate flomo
git clone https://github.com/pairlab/flomo.git && cd flomo

# 1. Backbone code. FloMo imports these clones from the repo root.
git clone https://github.com/Wan-Video/Wan2.2.git
git clone https://github.com/dreamzero0/dreamzero.git

# 2. PyTorch (cu128 shown; Blackwell GPUs need CUDA 12.8+), then FloMo.
pip install "torch==2.10.*" "torchvision==0.25.*" --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# 3. Wan VAE and text-encoder weights (into ./wan_checkpoints).
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir ./wan_checkpoints
```

The Wan DiT and the umT5 tokenizer download automatically on first use.

**For preprocessing only** (to make motion tokens from your own videos), also install
the 3D trackers. Both clones go in the repo root:

```bash
git clone https://github.com/henry123-boy/SpaTrackerV2.git
git clone https://github.com/facebookresearch/vggt.git
pip install "numpy<2" kornia==0.8.1 moviepy==1.0.0 pycolmap==3.11.1 pyceres==2.4 \
    scikit-learn flow_vis jaxtyping \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@d3a577acf0a9ad7e513a1416449a07b6f47d967f"
```

Version notes:

- Keep `torchvision<=0.25`, because SpaTrackerV2 uses `torchvision.io.write_video`,
  which 0.26 removed.
- Keep `numpy<2` on the same line, because without it pip installs numpy 2.
- Do not install SpaTrackerV2's `requirements.txt`, because it pins an `xformers` that
  needs an older torch.

## Pretrained checkpoints

The three paper base models are on Hugging Face at
[shitijgovil/FloMo](https://huggingface.co/shitijgovil/FloMo). All three use the same
training data. They differ only in what they predict:

| File | Predicts |
|---|---|
| `ma/flomo_ma.pt` | motion tokens + action |
| `va/flomo_va.pt` | video + action |
| `vma/flomo_vma.pt` | video + motion tokens + action |

```bash
hf download shitijgovil/FloMo vma/flomo_vma.pt --local-dir ./flomo_ckpts
```

## Inference

`infer_example.py` takes camera images, a task string, and the current end-effector
pose. It prints a chunk of future actions with shape `(T, 14)`: for each of the two
arms, `x, y, z, yaw, pitch, roll, gripper`.

```bash
python infer_example.py --checkpoint ./flomo_ckpts/vma/flomo_vma.pt \
    --wan-ckpt-dir ./wan_checkpoints \
    --image primary.png --image left.png --image right.png \
    --task "pick up the croissant and place it in the oven" \
    --ee-pose="x,y,z,yaw,pitch,roll,grip,x,y,z,yaw,pitch,roll,grip"
```

- Give one `--image` per camera, in the order `primary`, `left`, `right`.
- Write `--ee-pose=...` with `=`, because a negative first value otherwise looks like a flag.
- `--normalizer-type` selects the action scale: `realyam` (default) for the base
  checkpoints, `robotwin` for RoboTwin fine-tunes.

The script only predicts actions. Camera capture and robot control depend on your setup.

## Fine-tune on RoboTwin 2.0

This walk-through fine-tunes one RoboTwin task (`open_laptop`) from a base checkpoint.
It is the paper setup: 50 Aloha-AgileX `demo_clean` demos, 5,000 updates at batch
size 256. Preprocessing needs the tracker from [Install](#install).

**1. Download the demos.**

```bash
hf download TianxingChen/RoboTwin2.0 dataset/open_laptop/demo_clean.zip \
    --repo-type dataset --local-dir <RoboTwin>/data/hf
unzip <RoboTwin>/data/hf/dataset/open_laptop/demo_clean.zip -d <RoboTwin>/data/demo_clean
```

**2. Preprocess: 3D tracks and text embeddings.**

```bash
for mode in tracks text_emb; do
  python -m preprocessing.preprocess_robotwin mode=$mode task_names=[open_laptop] \
      source_dir=<RoboTwin>/data/demo_clean dest_dir=<RoboTwin>/data/flomo
done
```

**3. Fine-tune.** `resume=false` loads only the weights, so the optimizer and step
count start fresh.

```bash
python train.py --config-name train_robotwin data_root=<RoboTwin>/data \
    robotwin_task=open_laptop checkpoint=./flomo_ckpts/vma/flomo_vma.pt \
    resume=false run_name=robotwin_open_laptop
```

Checkpoints go to `checkpoints/forward_dynamics/robotwin_open_laptop/<epoch>.pt`.
VA and VMA print a warning about unused `video_primary` keys. This is expected,
because the RoboTwin config does not predict video.

**4. Run it.** Use [Inference](#inference) with `--normalizer-type robotwin`. To score
the policy, use RoboTwin's policy evaluator.

## Train on your own data

A dataset config in `cfg/datasets/` says which data to use and what each dataset
gives as input and output. `cfg/train.yaml` holds the training settings, and its
comments explain every key.

| `datasets=` | Use |
|---|---|
| `template` (default) | Annotated example. Copy it and fill in the `???` fields. |
| `flomo` | The paper mixture (EgoVerse + MolmoAct2 + RealYAM). |
| `robotwin` | One RoboTwin task (use `--config-name train_robotwin`). |

```bash
# Your own data
python train.py datasets=my_dataset run_name=my_run

# Start from a base checkpoint (weights only)
python train.py datasets=my_dataset checkpoint=./flomo_ckpts/vma/flomo_vma.pt resume=false run_name=my_ft

# Paper mixture, 8 GPUs on one node
torchrun --nproc_per_node=8 train.py datasets=flomo data_root=/path/to/flomo_data run_name=flomo_repro
```

- **Resume.** With `resume=true` (the default), a run with the same `run_name`
  continues from its last checkpoint.
- **Overrides.** You can set any config key on the command line, for example
  `optim.lr=5e-5 batch_size=64 gpu_max_bs=16`. `batch_size` is the global batch;
  `gpu_max_bs` is the per-GPU micro-batch.
- **Cache.** The first run computes motion statistics for each dataset and caches
  them in `~/.cache/flomo/track_stats`. Set `FLOMO_CACHE_DIR` to change the location.
- **Shared arguments in `flomo.yaml`.** The anchor `dataset_defaults` holds the
  shared dataset arguments. YAML expands it before command-line overrides, so
  `datasets.0.args.sample_stride=2` changes only the first dataset.

## Preprocessing reference

Each dataset has a script `preprocessing/preprocess_<dataset>.py` with a config
`cfg/preprocessing/preprocess_<dataset>.yaml`. Set `mode=` to run one stage:

| Dataset | Modes |
|---|---|
| RoboTwin | `tracks`, `text_emb`, `vae_latent_h16` |
| RealYAM | `tracks`, `text_emb`, `vae_latent_h16` |
| MolmoAct2 | `tracks`, `video`, `text`, `text_emb`, `vae_latent`, `vae_latent_h16`, `actions` |
| EgoVerse | `tracks`, `video`, `text`, `text_emb`, `meta`, `vae_latent_h16` |

- `tracks` makes the 3D tracks for motion tokens. `text_emb` caches the umT5 text
  embeddings that training reads.
- **EgoVerse order.** Run `video` before `tracks` and `vae_latent_h16`, and `text`
  before `text_emb`. `datasets=flomo` needs `video`, `tracks`, `vae_latent_h16`,
  `text`, and `text_emb`. In training, set `data_root` to the preprocessing
  `dest_dir`. The dataset names are `<output_name>_<split>`.
- **MolmoAct2 subsets.** Build an episode index, select a subset by hours, then
  preprocess the subset:

  ```bash
  python -m preprocessing.preprocess_molmoact2 index --index-dir /idx build --source-dir /raw
  python -m preprocessing.preprocess_molmoact2 index --index-dir /idx select --name subset --hours 10
  python -m preprocessing.preprocess_molmoact2 index_path=/idx/subset/index.jsonl mode=video
  ```

## Motion tokens

Motion tokens are dense 3D scene flow, encoded by the frozen Wan VAE as if it were
a 3-channel video:

1. **Track** (`preprocessing/SpatialTracker.py`). SpatialTrackerV2 tracks a 64×64
   grid of points (4,096 tracks) in 3D, in the camera frame of the window's first
   frame. It restarts every 16 frames.
2. **Load** (`BaseDataset.load_tracks`). With `supervision=cumulative_disp`, each
   16-frame window becomes the displacement from frame 0: `(1, 16, 4096, 3)`.
3. **Encode** (`MotionTokensWanPretrainedModality`). The modality makes an XYZ grid
   `(3, 15, 64, 64)`, resizes it to 256, scales it per dataset to `[-1, 1]`
   (`flomo/utils/motion_stats.py`), and encodes it with the Wan VAE into a
   `(48, 5, 16, 16)` latent. The Wan patch embedding turns the latent into tokens.

`supervision=point_tracks` and `supervision=optical_flow` encode per-frame track
changes instead. All datasets that output motion tokens must use the same mode.

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

- **Add a dataset.** Subclass `BaseDataset` and register it in
  `cotrain_loader._dataset_class`.
- **Add a modality.** Implement the `Modality` contract in `models/modalities.py`
  and add a `kind` branch to `_build_modality`.

## License

MIT. See [LICENSE](LICENSE).
