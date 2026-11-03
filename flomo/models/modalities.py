"""Per-modality encoding, trunk tokenization, flow matching, and per-sample losses.

`mask_and_encode` skips absent rows and fills their `[B, T_m, D] token slots
with a learned `[1, 1, D]` mask token owned by the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from flomo.utils.wan_encoding import encode_wan_vae


ROLE_INPUT = "input"
ROLE_OUTPUT = "output"
VALID_ROLES = (ROLE_INPUT, ROLE_OUTPUT)


def _comparison_video(gt: torch.Tensor, pred: torch.Tensor, src: Optional[torch.Tensor]):
    """Render `[src |] GT | prediction | error` from `(3, T, H, W)` RGB in [-1, 1]."""
    err = (gt - pred).abs()
    to_u8 = lambda x: ((x + 1.0) * 0.5 * 255.0).clamp(0, 255).to(torch.uint8)
    err_u8 = (err / err.max().clamp(min=1e-12) * 255.0).clamp(0, 255).to(torch.uint8)
    panels = (gt, pred) if src is None else (src, gt, pred)
    panel = torch.cat([to_u8(x) for x in panels] + [err_u8], dim=3)
    return rearrange(panel, "c t h w -> t c h w").cpu().numpy()


class Modality(nn.Module, ABC):
    """Abstract base. One instance per active modality; D is trunk-input dim."""

    def __init__(self, name: str, role: str, dim: int):
        super().__init__()
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
        self.name = str(name)
        self.role = role
        self.dim = int(dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.normal_(self.mask_token, std=0.02)
        # Assigned by the model; modalities may share a flow-matching scheduler.
        self.scheduler = None

    @abstractmethod
    def num_tokens(self) -> int:
        """T_m: trunk tokens this modality contributes per sample."""

    @abstractmethod
    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        """Preprocess raw inputs into latent space."""

    def encode_rows(self, raw: torch.Tensor, real_idx: torch.Tensor) -> torch.Tensor:
        """Encode selected batch rows; dataset-aware modalities use real_idx."""
        return self.encode(raw)

    @abstractmethod
    def tokenize(self, latent: torch.Tensor) -> torch.Tensor:
        """latent -> trunk tokens [B, T_m, D]."""

    def untokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """trunk tokens [B, T_m, D] -> latent. Output modalities override."""
        raise NotImplementedError(f"{type(self).__name__}: untokenize not implemented (input-only)")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent -> raw. Default identity; output modalities override if needed."""
        return latent

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample loss [B, ...]. Output modalities override."""
        raise NotImplementedError(f"{type(self).__name__}: compute_loss not implemented (input-only)")

    def sample_t(self, N: int, device: torch.device) -> torch.Tensor:
        """Sample N training timesteps from this modality's scheduler."""
        sched_ts = self.scheduler.timesteps.to(device)
        return sched_ts[torch.randint(0, len(sched_ts), (N,), device=device)]

    def add_noise(self, latent: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(noisy_latent, training_target)` from the same noise sample."""
        noise = torch.randn_like(latent)
        return (
            self.scheduler.add_noise(latent, noise, t),
            self.scheduler.training_target(latent, noise, t),
        )

    def init_noise(self, B: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """FM inference prior: N(0, I) at the modality's clean-latent shape.
        Output modalities override."""
        raise NotImplementedError(f"{type(self).__name__}: init_noise not implemented (input-only)")

    def viz_pred(self, batch, pred_latent: torch.Tensor):
        """Returns (video [T, 3, H, W] uint8, caption str | None) or None to skip."""
        return None

    @staticmethod
    def _caption(batch, i: int) -> Optional[str]:
        """Wandb caption for row `i`. `batch['text']` is always the raw string
        (embeddings live in `batch['text_emb']`)."""
        if "text" not in batch:  # e.g. infer_example passes only text_emb
            return None
        texts = batch["text"]
        item = texts[i]
        return item if isinstance(item, str) else None

    def gather_real(
        self,
        raw_modality_batch: List[Optional[torch.Tensor]],
        mask: torch.Tensor,
    ) -> "tuple[Optional[torch.Tensor], torch.Tensor]":
        """Stack real rows as `[N_real, *raw_shape]` and return their batch indices.

        Input is a length-B list of optional tensors; no real rows returns
        `(None, empty_idx)`.
        """
        B = mask.shape[0]
        if len(raw_modality_batch) != B:
            raise ValueError(
                f"{self.name}: raw_modality_batch has {len(raw_modality_batch)} entries but mask says B={B}"
            )
        real_idx = mask.nonzero(as_tuple=True)[0]                                       # [N_real]
        if real_idx.numel() == 0:
            return None, real_idx
        return torch.stack(
            [raw_modality_batch[i] for i in real_idx.tolist()], dim=0,
        ), real_idx                                                                     # [N_real, *raw_shape]

    def scatter_tokens(
        self,
        real_tokens: torch.Tensor,
        real_idx: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        """Scatter real-row tokens into a [B, T_m, D] buffer pre-filled with mask_token.
        `real_tokens` must have shape [real_idx.numel(), T_m, D] (or be empty when N_real=0).
        Callers may skip this entirely when real_idx covers all B rows."""
        T_m = self.num_tokens()
        out = self.mask_token.expand(B, T_m, self.dim).clone()
        if real_idx.numel() == 0:
            return out
        out = out.to(real_tokens.dtype)
        out[real_idx] = real_tokens
        return out

    def mask_and_encode(
        self,
        raw_modality_batch: List[Optional[torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode real rows and fill absent rows with mask_token in `[B, T_m, D]`.

        Input is a length-B list of optional tensors and a `[B]` bool mask.
        """
        real_modality_batch, real_idx = self.gather_real(raw_modality_batch, mask)
        B = mask.shape[0]
        if real_modality_batch is None:
            return self.scatter_tokens(
                torch.empty(0, self.num_tokens(), self.dim, device=mask.device),
                real_idx, B,
            )
        tokens = self.tokenize(self.encode_rows(real_modality_batch, real_idx))         # [N_real, T_m, D]
        if int(real_idx.numel()) == B:
            return tokens                                                                # all-real: no scatter needed
        return self.scatter_tokens(tokens, real_idx, B)


class _WanVAELatentModality(Modality):
    """Wan-VAE latents `(48, T_m_frames, H_lat, W_lat)` per sample.

    Tokenization uses Wan's `(1, 2, 2)` patch projection and unpatchify layout
    for pretrained weight sharing. Position encoding is native Wan RoPE.
    """

    PATCH_T = 1
    PATCH_H = 2
    PATCH_W = 2
    LATENT_CHANNELS = 48

    def __init__(
        self, name: str, role: str, dim: int, t_frames: int, h_lat: int, w_lat: int,
        view: Optional[str] = None,
    ):
        super().__init__(name=name, role=role, dim=dim)
        self.t_frames = int(t_frames)
        self.h_lat = int(h_lat)
        self.w_lat = int(w_lat)
        # Camera streams add learned view embeddings; other streams use view=None.
        self.view = view
        if self.h_lat % self.PATCH_H != 0 or self.w_lat % self.PATCH_W != 0:
            raise ValueError(
                f"{name}: latent grid ({self.h_lat}x{self.w_lat}) must be divisible "
                f"by patch ({self.PATCH_H}x{self.PATCH_W})"
            )
        self.h_tok = self.h_lat // self.PATCH_H
        self.w_tok = self.w_lat // self.PATCH_W
        patch_size = (self.PATCH_T, self.PATCH_H, self.PATCH_W)
        # FloMo replaces these projections with the shared pretrained Wan modules.
        self.patch_embedding = nn.Conv3d(
            self.LATENT_CHANNELS, self.dim, kernel_size=patch_size, stride=patch_size,
        )
        self.unpatch_proj = nn.Linear(self.dim, self.LATENT_CHANNELS * self.PATCH_T * self.PATCH_H * self.PATCH_W)
        self.vae: Optional[nn.Module] = None
        self.vae_spatial: Optional[int] = None
        self.use_preprocessed: bool = True
        if view is not None:
            self.view_pe = nn.Parameter(
                torch.zeros(1, self.LATENT_CHANNELS, self.t_frames, self.h_lat, self.w_lat)
            )
            nn.init.normal_(self.view_pe, std=0.02)

    def num_tokens(self) -> int:
        return self.t_frames * self.h_tok * self.w_tok

    @torch.no_grad()
    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        # preprocessed: raw is Wan-VAE latent (B, 48, t_frames, h_lat, w_lat) — identity.
        # live:         raw is pixels (B, H, W, 3) or (B, T, H, W, 3) — VAE-encode.
        if self.use_preprocessed:
            return raw
        s = self.vae_spatial
        return encode_wan_vae(self.vae, raw, (s, s), raw.device, torch.bfloat16)

    def tokenize(self, latent: torch.Tensor) -> torch.Tensor:
        # (B, 48, T, H, W) --Conv3d--> (B, D, T, h_tok, w_tok) --flatten--> (B, T*h_tok*w_tok, D)
        # Camera streams add the learned per-view PE to the latent first.
        if self.view is not None:
            latent = latent + self.view_pe.to(latent.dtype)
        x = self.patch_embedding(latent)
        return rearrange(x, "b d t h w -> b (t h w) d")

    def untokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        # (B, N, D) --Linear--> (B, N, 192) --unpatchify--> (B, 48, T, H, W).
        # The `(pt ph pw c)` ordering with c-fastest matches Wan's unpatchify einsum
        # `'fhwpqrc->cfphqwr'` so `unpatch_proj` weights map 1:1 to Wan's Head.head.
        x = self.unpatch_proj(tokens)
        return rearrange(
            x,
            "b (t ht wt) (pt ph pw c) -> b c (t pt) (ht ph) (wt pw)",
            t=self.t_frames, ht=self.h_tok, wt=self.w_tok,
            pt=self.PATCH_T, ph=self.PATCH_H, pw=self.PATCH_W, c=self.LATENT_CHANNELS,
        )

    def init_noise(self, B: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(
            B, self.LATENT_CHANNELS, self.t_frames, self.h_lat, self.w_lat,
            device=device, dtype=dtype,
        )

    @torch.no_grad()
    def viz_pred(self, batch, pred_latent: torch.Tensor):
        """GT|pred|err RGB panel via VAE decode for the first real row."""
        mask = batch[f"{self.name}_mask"]
        real_idx = mask.nonzero(as_tuple=True)[0]
        if real_idx.numel() == 0:
            return None
        i = int(real_idx[0])
        raw = batch[self.name][i].unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gt_latent = self.encode(raw)
            gt_rgb = self.vae.single_decode(gt_latent.to(torch.bfloat16))                # (1, 3, T, S, S) in [-1, 1]
            pred_rgb = self.vae.single_decode(pred_latent[i:i + 1].to(torch.bfloat16))
        video = _comparison_video(gt_rgb[0].float(), pred_rgb[0].float(), None)
        return video, self._caption(batch, i)


class ImageModality(_WanVAELatentModality):
    """Input-only single-frame conditioning at frame 0.

    Raw forms:
      preprocessed: `(B, 48, 1, H_lat, W_lat)`
      live: `(B, H_pix, W_pix, 3)`
    """

    def __init__(
        self, name: str, dim: int, h_lat: int, w_lat: int,
        view: Optional[str] = None,
    ):
        super().__init__(
            name=name, role=ROLE_INPUT, dim=dim, t_frames=1,
            h_lat=h_lat, w_lat=w_lat, view=view,
        )


class VideoModality(_WanVAELatentModality):
    """Causal video latents: 17 pixel frames encode to 5 Wan latents.

    Inputs keep all 5 latents. Outputs keep indices 1..4; the image modality
    supplies conditioning latent 0. Live inputs are `(B, T_pix, H, W, 3)`;
    preprocessed inputs are `(B, 48, t_frames, h_lat, w_lat)`, sliced by role.
    Output loss is flow-matching MSE over the target latents.
    """

    T_FRAMES_INPUT = 5
    T_FRAMES_OUTPUT = 4

    def __init__(
        self, name: str, role: str, dim: int, t_frames: int, h_lat: int, w_lat: int,
        view: Optional[str] = None,
    ):
        expected = self.T_FRAMES_INPUT if role == ROLE_INPUT else self.T_FRAMES_OUTPUT
        if t_frames != expected:
            raise ValueError(
                f"{name}: video as {role!r} requires t_frames={expected} "
                f"(input keeps the cond latent, output drops it), got t_frames={t_frames}. "
                f"Set forward_dynamics.modalities.video.t_frames={expected}."
            )
        super().__init__(
            name=name, role=role, dim=dim, t_frames=t_frames,
            h_lat=h_lat, w_lat=w_lat, view=view,
        )

    @torch.no_grad()
    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        if self.use_preprocessed:
            return raw                                                                  # (B, 48, t_frames, h_lat, w_lat)
        # Live: raw is (B, T_pix, H, W, 3); Wan VAE causal-encodes to (B, 48, 5, h, w).

        s = self.vae_spatial
        latent = encode_wan_vae(self.vae, raw, (s, s), raw.device, torch.bfloat16)      # (B, 48, t_frames+1, h, w)
        out = latent if self.role == ROLE_INPUT else latent[:, :, 1:]                   # INPUT keeps cond latent 0

        if out.shape[2] != self.t_frames:
            raise ValueError(
                f"{self.name}: encode expected t_frames={self.t_frames} in latent, got {out.shape[2]}"
            )

        return out

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [B, 48, t_frames, h_lat, w_lat]; per-sample MSE
        return (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3, 4))            # [B]

    @torch.no_grad()
    def viz_pred(self, batch, pred_latent: torch.Tensor):
        """GT|pred|err RGB panel over (1 + 4*t_frames) pixel frames. Splices the
        index-0 cond-image latent (from the `img` modality) in front of the
        video latents so Wan VAE's causal decode is well-formed."""
        mask = batch[f"{self.name}_mask"]
        real_idx = mask.nonzero(as_tuple=True)[0]
        if real_idx.numel() == 0:
            return None
        i = int(real_idx[0])
        s = self.vae_spatial
        img_field = batch[f"img_{self.view}"][i].unsqueeze(0)
        video_field = batch[self.name][i].unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if img_field.shape[1] == self.LATENT_CHANNELS:
                img_latent = img_field                                                  # (1, 48, 1, h, w)
            else:
                raw_img = img_field if img_field.dim() == 5 else img_field.unsqueeze(1)
                img_latent = encode_wan_vae(
                    self.vae, raw_img, (s, s), img_field.device, torch.bfloat16,
                )[:, :, -1:]                                                            # most recent latent = video frame 0
            if self.use_preprocessed:
                gt_full = torch.cat(
                    [img_latent.to(pred_latent), video_field.to(pred_latent)], dim=2,
                )                                                                        # (1, 48, t_frames+1, h, w)
            else:
                gt_full = encode_wan_vae(self.vae, video_field, (s, s), video_field.device, torch.bfloat16)
            gt_rgb = self.vae.single_decode(gt_full.to(torch.bfloat16))                 # (1, 3, T_pix, S, S)
            pred_full = torch.cat(
                [img_latent.to(pred_latent), pred_latent[i:i + 1]], dim=2,
            )                                                                            # (1, 48, t_frames+1, h, w)
            pred_rgb = self.vae.single_decode(pred_full.to(torch.bfloat16))
        video = _comparison_video(gt_rgb[0].float(), pred_rgb[0].float(), None)
        return video, self._caption(batch, i)


class _SequenceLinearModality(Modality):
    """Project `(B, T_m, feat_dim)` to `(B, T_m, D)` with learned position embeddings."""

    def __init__(self, name: str, role: str, dim: int, t_steps: int, feat_dim: int):
        super().__init__(name=name, role=role, dim=dim)
        self.t_steps = int(t_steps)
        self.feat_dim = int(feat_dim)
        self.proj = nn.Linear(self.feat_dim, self.dim)
        self.unproj = nn.Linear(self.dim, self.feat_dim)
        self.pe = nn.Parameter(torch.zeros(1, self.t_steps, self.dim))
        nn.init.normal_(self.pe, std=0.02)

    def num_tokens(self) -> int:
        return self.t_steps

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.dim() != 3 or raw.shape[1] != self.t_steps or raw.shape[2] != self.feat_dim:
            raise ValueError(
                f"{self.name}: encode expected (B, {self.t_steps}, {self.feat_dim}), "
                f"got {tuple(raw.shape)}"
            )
        return raw                                                           

    def tokenize(self, latent: torch.Tensor) -> torch.Tensor:
        return self.proj(latent) + self.pe # [B, T_m, D]

    def untokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.unproj(tokens) # [B, T_m, feat_dim]

    def init_noise(self, B: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(B, self.t_steps, self.feat_dim, device=device, dtype=dtype)


class ActionModality(_SequenceLinearModality):
    """Action output with flow-matching noise, targets, and MSE in raw action space."""

    def __init__(self, name: str, dim: int, t_steps: int, action_dim: int):
        super().__init__(name=name, role=ROLE_OUTPUT, dim=dim, t_steps=t_steps, feat_dim=action_dim)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [B, T_a, action_dim]; per-sample MSE in raw action space
        return (pred.float() - target.float()).pow(2).mean(dim=(1, 2))                  # [B]


class MotionTokensWanPretrainedModality(_WanVAELatentModality):
    """Motion grids encoded by the pretrained Wan VAE, with native RoPE only.

    `_traj_to_flow_grid` handles supervision-specific normalization.
    Preprocessed inputs `(B, 48, t_frames, h_lat, w_lat)` pass through unchanged.
    """

    def __init__(
        self,
        name: str,
        role: str,
        dim: int,
        t_frames: int,
        h_lat: int,
        w_lat: int,
        supervision: str = "cumulative_disp",
        tracks_ramen_normalized: bool = False,
    ):
        super().__init__(name=name, role=role, dim=dim, t_frames=t_frames, h_lat=h_lat, w_lat=w_lat)
        if supervision not in {"point_tracks", "optical_flow", "cumulative_disp"}:
            raise ValueError(
                f"{name}: supervision must be one of "
                f"{{point_tracks, optical_flow, cumulative_disp}}; got {supervision!r}"
            )
        if tracks_ramen_normalized and supervision != "cumulative_disp":
            raise ValueError(
                f"{name}: tracks_ramen_normalized=True requires supervision='cumulative_disp' "
                f"(got {supervision!r})"
            )
        self.supervision = supervision
        self.tracks_ramen_normalized = bool(tracks_ramen_normalized)
        # Per-dataset velocity bounds, shaped (num_ds, 3, 1, 1, 1).
        self.register_buffer("per_ds_flow_pct_low", torch.empty(0, 3, 1, 1, 1), persistent=True)
        self.register_buffer("per_ds_flow_pct_high", torch.empty(0, 3, 1, 1, 1), persistent=True)
        # Cumulative-displacement bounds for raw encoding and canonical display.
        self.register_buffer("per_ds_cumulative_disp_pct_low", torch.empty(0, 3, 1, 1, 1), persistent=True)
        self.register_buffer("per_ds_cumulative_disp_pct_high", torch.empty(0, 3, 1, 1, 1), persistent=True)
        # Per-dataset/channel/timestep Ramen statistics for physical reconstruction.
        self.register_buffer("per_ds_ramen_p02", torch.empty(0, 3, 0, 1, 1), persistent=True)
        self.register_buffer("per_ds_ramen_p98", torch.empty(0, 3, 0, 1, 1), persistent=True)
        self._ds_to_idx: Dict[str, int] = {}
        # Set by the model once per forward; consumed by encode via `ds_idx`.
        self._batch_ds_names: Optional[List[str]] = None

    def set_flow_pct_bounds_per_dataset(self, bounds: Dict[str, Tuple[Sequence[float], Sequence[float]]]) -> None:
        """Install per-dataset (pct_low, pct_high) bounds."""
        names = sorted(bounds.keys())
        los, his = [], []
        for n in names:
            lo, hi = bounds[n]
            lo_t = torch.as_tensor(list(lo), dtype=torch.float32).reshape(3, 1, 1, 1)
            hi_t = torch.as_tensor(list(hi), dtype=torch.float32).reshape(3, 1, 1, 1)
            if lo_t.numel() != 3 or hi_t.numel() != 3 or torch.any(hi_t <= lo_t):
                raise ValueError(f"{self.name}/{n}: per-DS bounds must be length-3 with high > low")
            los.append(lo_t)
            his.append(hi_t)
        self._ds_to_idx = {n: i for i, n in enumerate(names)}
        self.per_ds_flow_pct_low = torch.stack(los, dim=0).to(self.per_ds_flow_pct_low.device)
        self.per_ds_flow_pct_high = torch.stack(his, dim=0).to(self.per_ds_flow_pct_high.device)

    def set_cumulative_disp_pct_bounds_per_dataset(self, bounds: Dict[str, Tuple[Sequence[float], Sequence[float]]]) -> None:
        """Set cumulative-displacement bounds with keys matching `_ds_to_idx`."""
        if not self._ds_to_idx:
            raise RuntimeError(
                f"{self.name}: call set_flow_pct_bounds_per_dataset first; _ds_to_idx empty"
            )
        if set(bounds.keys()) != set(self._ds_to_idx.keys()):
            raise ValueError(
                f"{self.name}: cumulative_disp bounds keys {sorted(bounds.keys())} != "
                f"flow_pct keys {sorted(self._ds_to_idx.keys())}"
            )
        los, his = [], []
        for n in sorted(self._ds_to_idx.keys(), key=lambda k: self._ds_to_idx[k]):
            lo, hi = bounds[n]
            lo_t = torch.as_tensor(list(lo), dtype=torch.float32).reshape(3, 1, 1, 1)
            hi_t = torch.as_tensor(list(hi), dtype=torch.float32).reshape(3, 1, 1, 1)
            if lo_t.numel() != 3 or hi_t.numel() != 3 or torch.any(hi_t <= lo_t):
                raise ValueError(f"{self.name}/{n}: cumulative_disp bounds must be length-3 with high > low")
            los.append(lo_t)
            his.append(hi_t)
        self.per_ds_cumulative_disp_pct_low = torch.stack(los, dim=0).to(self.per_ds_cumulative_disp_pct_low.device)
        self.per_ds_cumulative_disp_pct_high = torch.stack(his, dim=0).to(self.per_ds_cumulative_disp_pct_high.device)

    def set_ramen_stats_per_dataset(self, stats: Dict[str, Tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]]) -> None:
        """Set Ramen `(H, 3)` statistics with keys matching `_ds_to_idx`.

        Store as `(num_ds, 3, H, 1, 1)` for broadcasting over flow grids.
        """
        if not self._ds_to_idx:
            raise RuntimeError(
                f"{self.name}: call set_flow_pct_bounds_per_dataset first; _ds_to_idx empty"
            )
        if set(stats.keys()) != set(self._ds_to_idx.keys()):
            raise ValueError(
                f"{self.name}: ramen stats keys {sorted(stats.keys())} != "
                f"flow_pct keys {sorted(self._ds_to_idx.keys())}"
            )
        p02s, p98s = [], []
        for n in sorted(self._ds_to_idx.keys(), key=lambda k: self._ds_to_idx[k]):
            p02, p98 = stats[n]
            p02_t = torch.as_tensor(p02, dtype=torch.float32)                     # (H, 3)
            p98_t = torch.as_tensor(p98, dtype=torch.float32)
            if p02_t.ndim != 2 or p02_t.shape[1] != 3 or p02_t.shape != p98_t.shape:
                raise ValueError(f"{self.name}/{n}: ramen stats must be (H, 3); got {tuple(p02_t.shape)}")
            # (H, 3) -> (3, H, 1, 1) for broadcasting across (3, T, h, w) flow grids.
            p02s.append(rearrange(p02_t, "h c -> c h 1 1"))
            p98s.append(rearrange(p98_t, "h c -> c h 1 1"))
        self.per_ds_ramen_p02 = torch.stack(p02s, dim=0).to(self.per_ds_ramen_p02.device)
        self.per_ds_ramen_p98 = torch.stack(p98s, dim=0).to(self.per_ds_ramen_p98.device)

    def _ds_idx_for(self, real_idx: torch.Tensor) -> Optional[torch.Tensor]:
        # Ramen inputs are normalized by the loader; other modes need dataset bounds.
        if self.supervision == "cumulative_disp" and self.tracks_ramen_normalized:
            return None
        assert self._ds_to_idx, f"{self.name}: per-DS bounds not installed"
        assert self._batch_ds_names is not None, f"{self.name}: _batch_ds_names not set; model must stash batch['dataset_name'] before encode"
        return torch.tensor(
            [self._ds_to_idx[self._batch_ds_names[i]] for i in real_idx.tolist()],
            device=self.per_ds_flow_pct_low.device, dtype=torch.long,
        )

    def encode_rows(self, raw: torch.Tensor, real_idx: torch.Tensor) -> torch.Tensor:
        return self.encode(raw, self._ds_idx_for(real_idx))

    def _spatial_resize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """`(B, 3, T_vel, side, side)` -> `(B, 3, T_vel, vae_spatial, vae_spatial)`
        via bilinear interp. No-op when sides already match."""
        s = self.vae_spatial
        side = flow.shape[-1]
        if s == side:
            return flow
        T_vel = flow.shape[2]
        flow = rearrange(flow, "b c t h w -> (b t) c h w")
        flow = F.interpolate(flow, size=(s, s), mode="bilinear", align_corners=False)
        return rearrange(flow, "(b t) c h w -> b c t h w", t=T_vel)

    def _traj_to_flow_grid(
        self, raw: torch.Tensor, ds_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Map `(B, 1, H, N, 3)` tracks to `(B, 3, H-1, vae_spatial, vae_spatial)`.

        Cumulative displacement drops the zero anchor. Ramen inputs are
        already normalized; raw displacements use cumulative_disp_pct bounds.
        Point tracks and optical flow use frame differences and flow_pct bounds.
        """
        if raw.ndim != 5 or raw.shape[-1] != 3:
            raise ValueError(f"{self.name}: expected (B, V, H, N, 3) tracks, got {tuple(raw.shape)}")
        B, V, H, N, _ = raw.shape
        if V != 1:
            raise ValueError(f"{self.name}: requires V=1, got V={V}")
        side = int(N ** 0.5)
        if side * side != N:
            raise ValueError(f"{self.name}: N={N} must be a perfect square")

        if self.supervision == "cumulative_disp":
            cd = raw[:, 0, 1:]                                                          # (B, T_vel, N, 3)
            flow = rearrange(cd, "b t (w h) d -> b d t h w", h=side, w=side)
            flow = self._spatial_resize_flow(flow)
            if self.tracks_ramen_normalized:
                return flow
            if ds_idx is None:
                raise RuntimeError(
                    f"{self.name}: ds_idx is required for cumulative_disp+no_norm (cumulative_disp_pct)"
                )
            lo = self.per_ds_cumulative_disp_pct_low.index_select(0, ds_idx).to(flow)         # (N_real, 3, 1, 1, 1)
            hi = self.per_ds_cumulative_disp_pct_high.index_select(0, ds_idx).to(flow)
            return torch.clamp(2.0 * (flow - lo) / (hi - lo) - 1.0, -1.0, 1.0)

        if ds_idx is None:
            raise RuntimeError(
                f"{self.name}: ds_idx is required for the flow_pct path "
                f"(supervision={self.supervision})"
            )
        tracks = raw[:, 0]                                                              # (B, H, N, 3)
        vel = tracks[:, 1:] - tracks[:, :-1]                                            # (B, T_vel, N, 3)
        flow = rearrange(vel, "b t (w h) d -> b d t h w", h=side, w=side)
        flow = self._spatial_resize_flow(flow)
        lo = self.per_ds_flow_pct_low.index_select(0, ds_idx).to(flow)                  # (N_real, 3, 1, 1, 1)
        hi = self.per_ds_flow_pct_high.index_select(0, ds_idx).to(flow)
        return torch.clamp(2.0 * (flow - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    @torch.no_grad()
    def encode(self, raw: torch.Tensor, ds_idx: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_preprocessed:
            return raw                                                                  # (B, 48, t_frames, h_lat, w_lat)
        flow_n = self._traj_to_flow_grid(raw, ds_idx)                                   # (B, 3, T_vel, s, s)
        T_vel = flow_n.shape[2]
        ts = 2 ** sum(self.vae.model.temperal_downsample)
        T_padded = 1 + ((T_vel - 1 + ts - 1) // ts) * ts
        if T_padded > T_vel:
            flow_n = F.pad(flow_n, (0, 0, 0, 0, 0, T_padded - T_vel))                   # zero-pad tail
        return self.vae.single_encode(flow_n.to(torch.bfloat16))                        # (B, 48, T_lat, h_lat, w_lat)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3, 4))            # [B]

    @torch.no_grad()
    def _decode_cond_img_3thw(self, batch, i: int) -> torch.Tensor:
        """Decode the single conditioning frame as `(3, 1, S, S)` float in [-1, 1].

        Inputs may be `(48, 1, h_lat, w_lat)` latents or `(H, W, 3)` / `(1, H, W, 3)` pixels.
        Encode live pixels first so visualization matches model conditioning.
        """
        img_raw = batch["img_primary"][i].unsqueeze(0)
        s = self.vae_spatial
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if img_raw.ndim in (4, 5) and img_raw.shape[-1] == 3:
                latent = encode_wan_vae(self.vae, img_raw, (s, s), img_raw.device, torch.bfloat16)
                rgb = self.vae.single_decode(latent.to(torch.bfloat16))                  # (1, 3, T_pix, S, S)
            elif img_raw.ndim == 5:                                                       # preprocessed latent: (1, 48, T, h_lat, w_lat)
                rgb = self.vae.single_decode(img_raw.to(torch.bfloat16))                 # (1, 3, T_pix, S, S)
            else:
                raise ValueError(f"{self.name}.viz: unexpected img shape {tuple(img_raw.shape)}")
        return rgb[0].float()

    @torch.no_grad()
    def _src_pixel_video(self, batch, i: int, T_vel: int, S: int) -> torch.Tensor:
        """Left-column source video (3, T_vel, S, S) in [-1, 1] from batch['viz_video'].
        Callers guard `viz_video` presence — no fallback here."""
        frames = batch["viz_video"][i]                                                    # (T_pix, H, W, 3) in [0, 255]
        pix = rearrange(frames[:T_vel].float() / 255.0 * 2.0 - 1.0, "t h w c -> c t h w")
        if pix.shape[-1] != S:
            pix = torch.nn.functional.interpolate(
                rearrange(pix, "c t h w -> t c h w"), size=(S, S), mode="bilinear", align_corners=False,
            )
            pix = rearrange(pix, "t c h w -> c t h w")
        return pix

    @torch.no_grad()
    def _decode_pair(self, batch, pred_latent: torch.Tensor):
        """Shared setup for viz_pred / viz_pred_normalized.

        Returns `(gt, pred, T_vel, S, i, ds_idx_int)` where `gt`/`pred` are
        `(3, T_vel, S, S)` in VAE-output space (= loss-target space). `None` if
        nothing to render.
        """
        mask = batch[f"{self.name}_mask"]
        real_idx = mask.nonzero(as_tuple=True)[0]
        if real_idx.numel() == 0:
            return None
        i = int(real_idx[0])
        if "viz_video" not in batch or batch["viz_video"][i] is None:
            return None
        raw = batch[self.name][i].unsqueeze(0)
        T_vel = raw.shape[2] - 1
        ds_idx_int = self._ds_to_idx[batch["dataset_name"][i]]
        skip_ds_idx = self.supervision == "cumulative_disp" and self.tracks_ramen_normalized
        ds_idx = None if skip_ds_idx else torch.tensor(
            [ds_idx_int], device=raw.device, dtype=torch.long,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gt_rgb = self.vae.single_decode(self.encode(raw, ds_idx).to(torch.bfloat16))  # (1, 3, T_padded, S, S)
            pred_rgb = self.vae.single_decode(pred_latent[i:i + 1].to(torch.bfloat16))
        gt = gt_rgb[0, :, :T_vel].float()                                                  # (3, T_vel, S, S)
        pred = pred_rgb[0, :, :T_vel].float()
        return gt, pred, T_vel, gt.shape[-1], i, ds_idx_int

    def _un_ramen(self, y: torch.Tensor, ds_idx_int: int, t_lo: int, t_hi: int) -> torch.Tensor:
        """Inverse of ramen normalize on a flow grid `(3, T, h, w)`. Maps
        `y` ∈ [-1, 1] back to physical cumulative_disp via per-(c, t) p02/p98 stats
        from `per_ds_ramen_{p02,p98}[ds_idx_int, :, t_lo:t_hi]`."""
        p02 = self.per_ds_ramen_p02[ds_idx_int, :, t_lo:t_hi].to(y)                       # (3, T, 1, 1)
        p98 = self.per_ds_ramen_p98[ds_idx_int, :, t_lo:t_hi].to(y)
        rng = (p98 - p02).clamp(min=1e-8)
        return (0.75 * y + 0.5) * rng + p02

    def _apply_cumulative_disp_pct(self, x: torch.Tensor, ds_idx_int: int) -> torch.Tensor:
        """Map physical cumulative_disp `(3, T, h, w)` to canonical [-1, 1] via per-DS
        cumulative_disp p0.01/p99.0 stats. Mirrors the velocity flow_pct mapping."""
        lo = self.per_ds_cumulative_disp_pct_low[ds_idx_int].to(x)                                # (3, 1, 1, 1)
        hi = self.per_ds_cumulative_disp_pct_high[ds_idx_int].to(x)
        return (2.0 * (x - lo) / (hi - lo) - 1.0).clamp(-1.0, 1.0)

    @torch.no_grad()
    def decode_positions(self, pred_latent: torch.Tensor, ds_idx: torch.Tensor, init_pts: torch.Tensor, T_vel: int):
        """Decode physical positions `(B, 1, T_vel+1, N, 3)` anchored at init_pts.

        Invert flow_pct then cumulatively sum velocities for optical_flow/point_tracks.
        For cumulative_disp, invert Ramen or cumulative_disp_pct normalization
        and add the resulting displacement directly to init_pts.
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out_norm = self.vae.single_decode(pred_latent.to(torch.bfloat16))                # (B, 3, T_pad, S, S)
        out_norm = out_norm.float().clamp(-1.0, 1.0)[:, :, :T_vel]
        side = int(round(init_pts.shape[3] ** 0.5))

        if self.supervision == "cumulative_disp":
            if self.tracks_ramen_normalized:
                phys = torch.stack([
                    self._un_ramen(out_norm[b], int(ds_idx[b]), t_lo=1, t_hi=1 + T_vel)
                    for b in range(out_norm.shape[0])
                ], dim=0)
            else:
                lo = self.per_ds_cumulative_disp_pct_low.index_select(0, ds_idx).to(out_norm)
                hi = self.per_ds_cumulative_disp_pct_high.index_select(0, ds_idx).to(out_norm)
                phys = (out_norm + 1.0) * 0.5 * (hi - lo) + lo
        else:
            lo = self.per_ds_flow_pct_low.index_select(0, ds_idx).to(out_norm)               # (B, 3, 1, 1, 1)
            hi = self.per_ds_flow_pct_high.index_select(0, ds_idx).to(out_norm)
            phys = (out_norm + 1.0) * 0.5 * (hi - lo) + lo

        if phys.shape[4] != side:
            phys = rearrange(phys, "b c t h w -> (b t) c h w")
            phys = torch.nn.functional.interpolate(phys, size=(side, side), mode="bilinear", align_corners=False)
            phys = rearrange(phys, "(b t) c h w -> b c t h w", t=T_vel)
        step = rearrange(phys, "b d t h w -> b 1 t (w h) d")                               # (B, 1, T_vel, N, 3)
        if self.supervision == "cumulative_disp":
            return torch.cat([init_pts, init_pts + step], dim=2)                             # (B, 1, T_vel+1, N, 3)
        return torch.cat([init_pts, init_pts + torch.cumsum(step, dim=2)], dim=2)

    def _render_flow_panel(self, batch, i: int, gt: torch.Tensor, pred: torch.Tensor):
        """Render display-space grids `(3, T, S, S)` as src | GT | pred | error."""
        src = self._src_pixel_video(batch, i, gt.shape[1], gt.shape[3])
        video = _comparison_video(gt, pred, src)
        return video, self._caption(batch, i)

    @torch.no_grad()
    def viz_pred(self, batch, pred_latent: torch.Tensor):
        """Render `src | GT | pred | error` in canonical display space.

        Ramen cumulative displacement is un-normalized to physical units, then
        mapped through cumulative_disp_pct. Other modes already decode in
        display space: cumulative_disp_pct for raw displacement, flow_pct for
        optical_flow/point_tracks.
        """
        out = self._decode_pair(batch, pred_latent)
        if out is None:
            return None
        gt, pred, T_vel, _, i, ds_idx_int = out

        if self.supervision == "cumulative_disp" and self.tracks_ramen_normalized:
            gt_phys = self._un_ramen(gt, ds_idx_int, t_lo=1, t_hi=1 + T_vel)
            pred_phys = self._un_ramen(pred, ds_idx_int, t_lo=1, t_hi=1 + T_vel)
            gt_c = self._apply_cumulative_disp_pct(gt_phys, ds_idx_int)
            pred_c = self._apply_cumulative_disp_pct(pred_phys, ds_idx_int)
        else:
            gt_c = gt
            pred_c = pred

        return self._render_flow_panel(batch, i, gt_c, pred_c)

    @torch.no_grad()
    def viz_pred_normalized(self, batch, pred_latent: torch.Tensor):
        """Render loss-target space; skip raw cumulative_disp, which matches viz_pred."""
        if self.supervision == "cumulative_disp" and not self.tracks_ramen_normalized:
            return None
        out = self._decode_pair(batch, pred_latent)
        if out is None:
            return None
        gt, pred, _, _, i, _ = out
        return self._render_flow_panel(batch, i, gt, pred)

    VIS_GRID = 24                                                                        # 24x24 display subsample

    @torch.no_grad()
    def viz_tracks(self, batch, pred_latent: torch.Tensor):
        """Overlay GT/predicted tracks on conditioning, subsampled to VIS_GRID².

        Tracks are in start_t's camera frame, so conditioning intrinsics suffice.
        Cumulative displacement uses batch['init_pts'] as its physical anchor;
        other modes use the first track frame. Include stationary points.
        """
        from flomo.utils.vis_utils import vis_pred as _vis_pred, _grid_subsample_indices
        mask = batch[f"{self.name}_mask"]
        real_idx = mask.nonzero(as_tuple=True)[0]
        if real_idx.numel() == 0:
            return None
        i = int(real_idx[0])
        if "intrs" not in batch or batch["intrs"][i] is None:
            return None
        if "viz_video" not in batch or batch["viz_video"][i] is None:
            return None

        raw_tracks = batch[self.name][i].unsqueeze(0)                                    # (1, V=1, T_vel+1, N, 3)
        T_vel = raw_tracks.shape[2] - 1
        device = raw_tracks.device

        if self.supervision == "cumulative_disp":
            if "init_pts" not in batch or batch["init_pts"][i] is None:
                return None
            init_raw = batch["init_pts"][i].to(device).float()                            # (V=1, 1, N, 3)
            init_pts = init_raw.unsqueeze(0)                                              # (1, V=1, 1, N, 3)
        else:
            init_pts = raw_tracks[:, :, 0:1]                                              # (1, 1, 1, N, 3)

        ds_idx = self._ds_to_idx[batch["dataset_name"][i]]
        pred_xyz = self.decode_positions(
            pred_latent[i:i + 1], torch.tensor([ds_idx], dtype=torch.long, device=device), init_pts, T_vel,
        )

        if self.supervision == "cumulative_disp":
            cd_gt = raw_tracks[:, :, 1:].float()                                          # (1, 1, T_vel, N, 3)
            if self.tracks_ramen_normalized:
                # Apply un-ramen pointwise: stats (3, T_vel) for original frames 1..H-1.
                p02 = rearrange(
                    self.per_ds_ramen_p02[ds_idx, :, 1:1 + T_vel, 0, 0], "c t -> 1 1 t 1 c"
                ).to(cd_gt)
                p98 = rearrange(
                    self.per_ds_ramen_p98[ds_idx, :, 1:1 + T_vel, 0, 0], "c t -> 1 1 t 1 c"
                ).to(cd_gt)
                rng = (p98 - p02).clamp(min=1e-8)
                cd_gt = (0.75 * cd_gt + 0.5) * rng + p02
            gt_xyz = torch.cat([init_pts, init_pts + cd_gt], dim=2)                       # (1, 1, T_vel+1, N, 3)
        else:
            gt_xyz = raw_tracks                                                            # (1, 1, T_vel+1, N, 3)
        # Project XYZ -> (row, col) using cond intrinsics (V=1).
        K = batch["intrs"][i].to(pred_xyz.device).float()                                # (V=1, 3, 3)
        cond_3thw = self._decode_cond_img_3thw(batch, i)                                 # (3, 1, S, S)
        src_chw = cond_3thw[:, 0]                                                        # (3, S, S)
        H_pix, W_pix = src_chw.shape[-2], src_chw.shape[-1]
        fx, fy, cx, cy = K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]
        def _proj(xyz):                                                                  # (1, V, T, N, 3) -> (1, V, T, N, 2) row, col in [-1, 1]
            X = xyz[..., 0]; Y = xyz[..., 1]; Z = xyz[..., 2].clamp(min=1e-3)
            u = fx * X / Z + cx
            v = fy * Y / Z + cy
            col_n = (u - W_pix / 2.0) / (W_pix / 2.0)
            row_n = (v - H_pix / 2.0) / (H_pix / 2.0)
            return torch.stack([row_n, col_n], dim=-1).clamp(-1.0, 1.0)
        gt_rc = _proj(gt_xyz)
        pred_rc = _proj(pred_xyz)
        sub = _grid_subsample_indices(gt_rc.shape[3], self.VIS_GRID)
        if sub is not None:
            sub_t = torch.as_tensor(sub, dtype=torch.long, device=gt_rc.device)
            gt_rc = gt_rc.index_select(dim=3, index=sub_t)
            pred_rc = pred_rc.index_select(dim=3, index=sub_t)
        # Overlay base: decoded cond frame at start_t, in [0, 1].
        src_rgb_01 = ((src_chw + 1.0) * 0.5).clamp(0, 1)
        src = rearrange(src_rgb_01, "c h w -> 1 1 h w c")                                # (1, 1, H, W, 3) in [0, 1]
        T = gt_rc.shape[2]
        raw = batch["viz_video"][i][:T].float() / 255.0                                  # (T, H, W, 3) in [0, 1]
        if raw.shape[1:3] != (H_pix, W_pix):
            raw = rearrange(raw, "t h w c -> t c h w")
            raw = torch.nn.functional.interpolate(raw, size=(H_pix, W_pix), mode="bilinear", align_corners=False)
            raw = rearrange(raw, "t c h w -> t h w c")
        vid_u8 = (raw.clamp(0, 1) * 255.0).to(torch.uint8).cpu()                         # (T, H, W, 3)
        frames = []
        for t in range(T):
            gt_ov = _vis_pred(src, gt_rc[:, :, : t + 1], cmap="cool", show_all=True)[0]    # (H, W, 3) uint8
            pred_ov = _vis_pred(src, pred_rc[:, :, : t + 1], cmap="autumn_r", show_all=True)[0]
            frames.append(torch.cat([vid_u8[t], gt_ov, pred_ov], dim=1))                   # (H, 3W, 3)
        video = rearrange(torch.stack(frames, dim=0), "t h w c -> t c h w").cpu().numpy()
        return video, self._caption(batch, i)


class TextModality(Modality):
    """Cross-attention text input; contributes no trunk tokens."""

    def __init__(self, name: str, dim: int, text_len: int, text_dim: int):
        super().__init__(name=name, role=ROLE_INPUT, dim=dim)
        self.text_len = int(text_len)
        self.text_dim = int(text_dim)
        self.text_encoder: Optional[nn.Module] = None
        self.tokenizer = None
        self.use_preprocessed: bool = True

    def num_tokens(self) -> int:
        return 0

    @torch.no_grad()
    def encode(self, raw):
        # preprocessed: raw is Tensor [N, text_len, text_dim] — identity.
        # live:         raw is List[str] of length N — UMT5 + zero pad.
        if self.use_preprocessed:
            return raw
        device = next(self.text_encoder.parameters()).device
        tok = self.tokenizer(
            raw, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.text_len,
        ).to(device)
        ctx = self.text_encoder(tok["input_ids"], tok["attention_mask"])                # [N, text_len, text_dim]
        return ctx * tok["attention_mask"].unsqueeze(-1).to(ctx.dtype)                  # zero pad rows

    def tokenize(self, latent: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            f"{self.name}: tokenize is not used. Text enters via cross-attention "
            f"context; pull it with encode_context()."
        )

    def encode_context(self, batch, mask: Optional[torch.Tensor], B: int) -> torch.Tensor:
        """Return `[B, text_len, text_dim]` context from text_emb or raw text.

        A `[B]` bool mask selects real rows; absent rows receive zeros.
        With neither text key present, return Wan's all-zero unconditional context.
        """
        if "text" not in batch and "text_emb" not in batch:
            return torch.zeros(
                B, self.text_len, self.text_dim, device=self.mask_token.device,
            )
        key = "text_emb" if self.use_preprocessed else "text"
        text_batch = batch[key]
        if len(text_batch) != B:
            raise ValueError(f"{self.name}: batch['{key}'] len={len(text_batch)} but mask says B={B}")
        real_idx = mask.nonzero(as_tuple=True)[0]
        if real_idx.numel() == 0:
            raise RuntimeError(
                f"{self.name}: cross-attention context absent on every row; text is required for now."
            )
        real_items = [text_batch[int(i)] for i in real_idx.tolist()]
        real_input = torch.stack(real_items, dim=0) if self.use_preprocessed else real_items
        real_encoded = self.encode(real_input)                                          # [N_real, text_len, text_dim]
        if int(real_idx.numel()) == B:
            return real_encoded
        out = real_encoded.new_zeros(B, real_encoded.shape[1], real_encoded.shape[2])
        out[real_idx] = real_encoded
        return out
