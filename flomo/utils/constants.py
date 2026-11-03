# Preprocessing
POSSIBLE_OUTPUTS = ['3d', '2d', 'depth', 'pose']
SPATIAL_TRACKER_OUTPUTS = ['3d', '2d', 'depth', 'pose']
VGGT_INPUT_SIZE=518
VGGT_OUTPUTS = ['pose', 'depth']
MULTI_VIEW_TRACKERS = ['spatialtracker', 'vggt']
TRACKS_FORMAT = {
    "spatialtracker": "rc",
    "vggt": "rc",
}

# Wan DiT backbone size presets. head_dim is implicit at dim // num_heads.
#
#   "5B"  : Wan 2.2 5B pretrained (ti2v_5B). Required for init_weights=pretrained.
#   "50M" : ~54M scratch model.
#   "20M" : ~22M scratch model; fits 24GB VRAM @ bs=32+.
WAN_MODEL_SIZES = {
    "5B":  {"dim": 3072, "num_heads": 24, "num_layers": 30, "ffn_dim": 14336},
    "50M": {"dim": 512,  "num_heads": 8,  "num_layers": 12, "ffn_dim": 2048},
    "20M": {"dim": 384,  "num_heads": 6,  "num_layers": 8,  "ffn_dim": 1536},
}

_WAN_SIZE_FIELDS = ("dim", "num_heads", "num_layers", "ffn_dim")


def resolve_wan_size(wan_cfg):
    """Merge size_preset with explicit dim/num_heads/num_layers/ffn_dim from cfg.

    Returns a dict with all four fields. Explicit cfg values take precedence;
    `null` keys fall back to the named preset. Every key must be present.
    size_preset=null requires all four to be set explicitly. Fails hard on
    unknown preset name or any unresolved field."""
    preset_name = wan_cfg.size_preset
    if preset_name is None:
        preset = {}
    elif preset_name not in WAN_MODEL_SIZES:
        raise ValueError(
            f"Unknown wan.size_preset '{preset_name}'. "
            f"Choices: {list(WAN_MODEL_SIZES)} or null."
        )
    else:
        preset = WAN_MODEL_SIZES[preset_name]

    out = {}
    for f in _WAN_SIZE_FIELDS:
        v = wan_cfg[f]
        if v is None and f in preset:
            v = preset[f]
        if v is None:
            raise ValueError(
                f"wan.{f} unresolved: not in cfg and not in "
                f"size_preset='{preset_name}'. Set wan.{f} explicitly or pick "
                f"a preset that defines it."
            )
        out[f] = v
    return out
