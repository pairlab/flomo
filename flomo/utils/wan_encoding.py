from einops import rearrange

def encode_wan_text(text_encoder, tokenizer, text, device, dtype):
    tok = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)
    attention_mask = tok.attention_mask.to(device)
    emb = text_encoder(input_ids, attention_mask)
    mask = rearrange(
        attention_mask.to(device=emb.device, dtype=emb.dtype),
        "b l -> b l 1",
    )
    emb = emb * mask
    return emb.to(dtype=dtype)


def encode_wan_vae(vae, pixels, image_hw, device, dtype):
    """Encode pixels through the Wan VAE.

    pixels   : (b, h, w, c) image or (b, t, h, w, c) video, uint8 or float.
               Must already be at image_hw using the dataset's exact LANCZOS
               preprocessing (direct squash for RealYam, 4:3 crop + squash for
               MolmoAct2, shortest-side resize + crop for EgoVerse).
    image_hw : target (h_px, w_px); fail-hard if pixels don't match.
    Returns  : latent (b, c, t, h_lat, w_lat) — always 5D (t=1 for image input)
    """
    if pixels.ndim == 4:
        x = rearrange(pixels, "b h w c -> b c 1 h w").float()
    elif pixels.ndim == 5:
        x = rearrange(pixels, "b t h w c -> b c t h w").float()
    else:
        raise ValueError(
            f"Expected (b, h, w, c) or (b, t, h, w, c), got {tuple(pixels.shape)}"
        )
    if x.max() > 1.0:
        x = x / 255.0
    x = x * 2.0 - 1.0
    h_px, w_px = image_hw
    if x.shape[-2] != h_px or x.shape[-1] != w_px:
        raise ValueError(
            f"encode_wan_vae: input pixels {x.shape[-2]}x{x.shape[-1]} must already "
            f"match image_hw={image_hw}. Apply the dataset's exact LANCZOS resize "
            f"at the loader level."
        )
    down = int(vae.upsampling_factor)
    if h_px % down != 0 or w_px % down != 0:
        raise ValueError(
            f"Wan VAE input shape {image_hw} must be divisible by upsampling_factor={down}."
        )
    h_lat = h_px // down
    w_lat = w_px // down
    latent = vae.single_encode(x.to(device=device, dtype=dtype))
    if latent.shape[-2] != h_lat or latent.shape[-1] != w_lat:
        raise ValueError(
            f"Expected Wan VAE latent spatial ({h_lat}, {w_lat}), "
            f"got {tuple(latent.shape)}."
        )
    return latent


def encode_wan_vae_video(vae, video, image_hw, device, dtype):
    """Encode a batch of B videos, each T frames, with the Wan 2.2 VAE.

    video: (b, t, h, w, c) uint8 or float. Returns latent
    (b, z, t_lat, h_lat, w_lat). The VAE compresses time 4x and consumes frames
    in 1+4k groups, so t must be one of {1, 5, 9, 13, 17, ...}; otherwise the
    encoder silently drops the trailing (t-1) % 4 frames. t_lat = 1 + (t-1)//4.
    """
    if video.ndim != 5:
        raise ValueError(f"Expected video shape (b, t, h, w, c), got {tuple(video.shape)}")
    t = video.shape[1]
    if (t - 1) % 4 != 0:
        raise ValueError(
            f"Wan VAE consumes frames in 1+4k groups; t={t} would silently drop "
            f"{(t - 1) % 4} frame(s). Pass t in {{1, 5, 9, 13, 17, ...}}."
        )
    vid = rearrange(video, "b t h w c -> b c t h w").float()
    if vid.max() > 1.0:
        vid = vid / 255.0
    vid = vid * 2.0 - 1.0
    h_px, w_px = image_hw
    if vid.shape[-2] != h_px or vid.shape[-1] != w_px:
        raise ValueError(
            f"encode_wan_vae_video: frames {vid.shape[-2]}x{vid.shape[-1]} must already "
            f"match image_hw={image_hw}. Apply the dataset's exact LANCZOS resize "
            f"at the loader level."
        )
    latent = vae.single_encode(vid.to(device=device, dtype=dtype))
    down = int(vae.upsampling_factor)
    if h_px % down != 0 or w_px % down != 0:
        raise ValueError(
            f"Wan VAE input shape {image_hw} must be divisible by upsampling_factor={down}."
        )
    h_lat = h_px // down
    w_lat = w_px // down
    t_lat = 1 + (t - 1) // 4
    if tuple(latent.shape[1:]) != (vae.z_dim, t_lat, h_lat, w_lat):
        raise ValueError(
            f"Expected Wan VAE latent shape (b, {vae.z_dim}, {t_lat}, {h_lat}, {w_lat}), "
            f"got {tuple(latent.shape)}."
        )
    return latent
