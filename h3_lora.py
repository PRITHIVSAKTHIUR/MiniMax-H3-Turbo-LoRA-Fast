"""Turbo and style LoRA support for the diffusers MiniMax-H3 transformer: several checkpoint layouts, one fold
mechanism.

Every LoRA is applied by folding `scale * (lora_B @ lora_A)` into the bf16 weights rather than as runtime
wrappers, because the AoTI block package (`h3_aoti`) reads each block's live weights and a wrapper module would
be invisible to it. Deltas are computed in float32 and round once on the way back into bf16. The low-rank factors
of every loaded LoRA stay resident, so the active one can be switched per request (`set_active`) — unfold the
old, fold the new, in place, through one bf16 rounding.

The supported checkpoints ship in different layouts:

* `larry` (`larryvrh/MiniMax-H3-Turbo-Lora`) targets the reference (ComfyUI) module tree —
  `blocks.N.attn.qkv_proj`, `blocks.N.mlp.fc1`, `token_refiner.blocks.N`, `final_layer.adaln_proj.linear` — with
  `alpha == rank` (scale 1). Each delta gets the same transform the base weights got in the diffusers conversion
  (scripts/convert_minimax_h3_to_diffusers.py, huggingface/diffusers#14371): fused-QKV row thirds onto
  `attn.to_q/k/v`, the `SwiGLU` gate/value swap onto `ff.net.0.proj`, `fc2` -> `ff.net.2`, `blocks.` ->
  `transformer_blocks.`, `token_refiner.blocks.` -> `token_refiner.refiner_blocks.`,
  `final_layer.adaln_proj.linear` -> `norm_out.linear`. The row transforms are applied to `lora_B` directly
  (rows of `B @ A` are rows of `B`), so no full delta is ever materialized at load.

* `lightx` / `lightx8` (`lightx2v/Minimax-h3-Turbo`) are PEFT checkpoints against the diffusers tree itself —
  `transformer_blocks.N.attn.to_q.lora_A.default.weight` and friends — rank 128, `alpha == 8`, so the fold scale
  is `8 / 128 = 0.0625` (matching `set_adapters(weights=1.0)` in their inference script). Keys map name-for-name.
  `lightx` is the 4-step file, `lightx8` the 8-step v1.0 file.

* `realism` (`fal/MiniMax-H3-Realism-People-LoRA`) is a style LoRA, not a turbo one — realistic people, trigger
  word `r34l1sm`. Same reference tree as larry under a `diffusion_model.` prefix, attention only (`qkv_proj` /
  `out_proj` on the 52 blocks), rank 16, `alpha == rank` (the card's scale 1.0), so it goes through the same
  `_larry_targets` mapping. It wants the full step count, not 4–6.

* `joyfox` (`joyfox/MiniMax-H3-Turbo`) is another 4-step turbo LoRA, ComfyUI-native like realism but covering
  more of the tree: attention, MLP, the block and final AdaLN projections, and both output heads
  (`final_layer.video_out` -> `proj_out`, `final_layer.audio_out` -> `audio_proj_out`). Ranks are mixed
  (32 attention/MLP, 8 modulation/heads) with a scalar `alpha` per key, so each `lora_B` is prescaled by
  `alpha / rank` at load (here always 1.0). Its `adaln_proj` entries are skipped: the Comfy-Org checkpoint it was
  trained against gives the modulation projection an 8-dimensional input (`[96768, 8]`), while the diffusers
  transformer projects the full 2688-dim time embedding (`[96768, 2688]`), so those deltas have no counterpart
  to fold into.

* `H3-Facial-Realism-CloseUp` (`prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp`) is a close-up facial realism
  style LoRA, not a turbo one — it wants the full step count, not 4–6. Its layout is detected at load time by
  `_load_auto`: PEFT keys against the diffusers tree map name-for-name, everything else is treated as the
  reference tree (with or without the `diffusion_model.` prefix) and goes through `_larry_targets`. Per-key
  scalar `alpha` values are folded into `lora_B` as `alpha / rank` when present, otherwise `alpha == rank` is
  assumed; reference-tree `adaln_proj` entries whose input dim is not the 2688-dim time embedding of the
  diffusers tree are skipped, like joyfox's.

* `H3-I2V-Anime-Motion` (`prithivMLmods/MiniMax-H3-I2V-Anime-Motion-LoRA`) is an anime motion style LoRA built
  for image-to-video, trigger word `Anime-Motion` — put the trigger word in the prompt and pair the set with a
  first frame for the intended effect. It wants the full step count, not 4–6, and is loaded through the same
  `_load_auto` layout detection as the facial LoRA.

`H3_LORA` selects the larry file (`off` skips loading it), `H3_LIGHTX=off` skips lightx, `H3_REALISM=off` skips
realism, `H3_JOYFOX=off` skips joyfox, `H3_LIGHTX8=off` skips the lightx 8-step file, `H3_FACIAL=off` skips the
facial realism LoRA, `H3_ANIME=off` skips the anime motion LoRA, `H3_LORA_DEFAULT` picks which set starts
folded, and `H3_LORA_STRENGTH` is the larry card's sharpness/artifact dial.
"""

from __future__ import annotations

import os

import torch

LARRY_REPO = os.environ.get("H3_LORA_REPO", "larryvrh/MiniMax-H3-Turbo-Lora")
LARRY_FILE = os.environ.get("H3_LORA", "minimax_h3_turbo_v4_step600_ema.safetensors")
LIGHTX_REPO = os.environ.get("H3_LIGHTX_REPO", "lightx2v/Minimax-h3-Turbo")
LIGHTX_FILE = os.environ.get("H3_LIGHTX_FILE", "minimax_h3_fl2v_turbo_4step_v0.1.safetensors")
LIGHTX8_FILE = os.environ.get("H3_LIGHTX8_FILE", "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors")
LIGHTX_ALPHA = 8
REALISM_REPO = os.environ.get("H3_REALISM_REPO", "fal/MiniMax-H3-Realism-People-LoRA")
REALISM_FILE = os.environ.get("H3_REALISM_FILE", "h3-realism-people-t2v-i2v-r2v.safetensors")
JOYFOX_REPO = os.environ.get("H3_JOYFOX_REPO", "joyfox/MiniMax-H3-Turbo")
JOYFOX_FILE = os.environ.get("H3_JOYFOX_FILE", "minimax_h3_fl2va_4step_lora.safetensors")
FACIAL_REPO = os.environ.get("H3_FACIAL_REPO", "prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp")
FACIAL_FILE = os.environ.get("H3_FACIAL_FILE", "minimax-h3-facial-realism-closeup-cp2000.safetensors")
FACIAL_NAME = "H3-Facial-Realism-CloseUp"
ANIME_REPO = os.environ.get("H3_ANIME_REPO", "prithivMLmods/MiniMax-H3-I2V-Anime-Motion-LoRA")
ANIME_FILE = os.environ.get("H3_ANIME_FILE", "MiniMax-H3-I2V-Anime-Motion-LoRA-1400.safetensors")
ANIME_NAME = "H3-I2V-Anime-Motion"
LARRY_STRENGTH = float(os.environ.get("H3_LORA_STRENGTH", "1.0"))
DEFAULT_LORA = os.environ.get("H3_LORA_DEFAULT", "larry")


def _larry_targets(name: str, b: torch.Tensor, inner_dim: int) -> list[tuple[str, torch.Tensor]]:
    """Map one reference-tree base name and its `lora_B` onto diffusers parameter key + row-transformed B."""
    if name.startswith("token_refiner.blocks."):
        target = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif name.startswith("blocks."):
        target = name.replace("blocks.", "transformer_blocks.", 1)
    else:
        target = name
    target = target.replace("final_layer.adaln_proj.linear", "norm_out.linear")
    target = target.replace("final_layer.video_out", "proj_out").replace("final_layer.audio_out", "audio_proj_out")

    if target.endswith(".attn.qkv_proj"):
        prefix = target.removesuffix("qkv_proj")
        return [
            (f"{prefix}to_{kind}.weight", part.contiguous())
            for kind, part in zip(("q", "k", "v"), b.split(inner_dim, dim=0))
        ]
    if target.endswith(".mlp.fc1"):
        gate, value = b.chunk(2, dim=0)
        return [(target.replace(".mlp.fc1", ".ff.net.0.proj") + ".weight", torch.cat([value, gate]).contiguous())]
    if target.endswith(".mlp.fc2"):
        return [(target.replace(".mlp.fc2", ".ff.net.2") + ".weight", b)]
    if target.endswith(".attn.out_proj"):
        return [(target.replace(".attn.out_proj", ".attn.to_out.0") + ".weight", b)]
    return [(target + ".weight", b)]


def _load_larry(inner_dim: int) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lora = load_file(hf_hub_download(LARRY_REPO, LARRY_FILE))
    bases = sorted({key.rsplit(".lora_", 1)[0] for key in lora})
    entries = []
    for name in bases:
        a = lora[f"{name}.lora_A.weight"]
        b = lora[f"{name}.lora_B.weight"]
        entries.extend((key, a, b_part) for key, b_part in _larry_targets(name, b, inner_dim))
    return {
        "label": f"{LARRY_REPO}/{LARRY_FILE}",
        "scale": LARRY_STRENGTH,
        "entries": entries,
    }


def _load_lightx(file: str = LIGHTX_FILE) -> dict:
    """A diffusers-native PEFT checkpoint from `lightx2v/Minimax-h3-Turbo` — keys map name-for-name."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lora = load_file(hf_hub_download(LIGHTX_REPO, file))
    suffix_a, suffix_b = ".lora_A.default.weight", ".lora_B.default.weight"
    bases = sorted({key[: -len(suffix_a)] for key in lora if key.endswith(suffix_a)})
    ranks = {lora[f"{name}{suffix_a}"].shape[0] for name in bases}
    if len(ranks) != 1:
        raise ValueError(f"Mixed LoRA ranks in {file}: {sorted(ranks)}")
    entries = [(f"{name}.weight", lora[f"{name}{suffix_a}"], lora[f"{name}{suffix_b}"]) for name in bases]
    return {
        "label": f"{LIGHTX_REPO}/{file}",
        "scale": LIGHTX_ALPHA / ranks.pop(),
        "entries": entries,
    }


def _load_realism(inner_dim: int) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lora = load_file(hf_hub_download(REALISM_REPO, REALISM_FILE))
    bases = sorted({key.rsplit(".lora_", 1)[0].removeprefix("diffusion_model.") for key in lora})
    entries = []
    for name in bases:
        a = lora[f"diffusion_model.{name}.lora_A.weight"]
        b = lora[f"diffusion_model.{name}.lora_B.weight"]
        entries.extend((key, a, b_part) for key, b_part in _larry_targets(name, b, inner_dim))
    return {
        "label": f"{REALISM_REPO}/{REALISM_FILE}",
        "scale": 1.0,
        "entries": entries,
    }


def _load_joyfox(inner_dim: int) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lora = load_file(hf_hub_download(JOYFOX_REPO, JOYFOX_FILE))
    bases = sorted({key.rsplit(".lora_", 1)[0].removeprefix("diffusion_model.") for key in lora if ".lora_" in key})
    entries = []
    for name in bases:
        if "adaln_proj" in name:
            continue
        prefixed = f"diffusion_model.{name}"
        a = lora[f"{prefixed}.lora_A.weight"]
        b = lora[f"{prefixed}.lora_B.weight"] * (lora[f"{prefixed}.alpha"].item() / a.shape[0])
        entries.extend((key, a, b_part) for key, b_part in _larry_targets(name, b, inner_dim))
    return {
        "label": f"{JOYFOX_REPO}/{JOYFOX_FILE}",
        "scale": 1.0,
        "entries": entries,
    }


def _load_auto(repo: str, file: str, inner_dim: int) -> dict:
    """Style LoRA whose checkpoint layout is detected at load time: PEFT keys against the diffusers tree map
    name-for-name, everything else is treated as the reference tree (with or without a `diffusion_model.`
    prefix) and goes through `_larry_targets`. Per-key scalar `alpha` is folded into `lora_B` as `alpha / rank`
    when present, otherwise `alpha == rank` is assumed; reference-tree `adaln_proj` entries whose input dim is
    not the 2688-dim time embedding of the diffusers tree are skipped, like joyfox's."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lora = load_file(hf_hub_download(repo, file))
    label = f"{repo}/{file}"

    suffix_a, suffix_b = ".lora_A.default.weight", ".lora_B.default.weight"
    if any(key.endswith(suffix_a) for key in lora):
        entries = []
        for base in sorted({key[: -len(suffix_a)] for key in lora if key.endswith(suffix_a)}):
            a = lora[f"{base}{suffix_a}"]
            b = lora[f"{base}{suffix_b}"]
            alpha = lora.get(f"{base}.alpha")
            if alpha is not None:
                b = b * (alpha.item() / a.shape[0])
            name = base.removeprefix("transformer.")
            entries.append((f"{name}.weight", a, b))
        return {"label": label, "scale": 1.0, "entries": entries}

    prefix = "diffusion_model." if any(key.startswith("diffusion_model.") for key in lora) else ""
    entries = []
    for name in sorted({key.rsplit(".lora_", 1)[0].removeprefix(prefix) for key in lora if ".lora_" in key}):
        full = f"{prefix}{name}"
        a = lora[f"{full}.lora_A.weight"]
        if "adaln_proj" in name and a.shape[1] != 2688:
            continue
        b = lora[f"{full}.lora_B.weight"]
        alpha = lora.get(f"{full}.alpha")
        if alpha is not None:
            b = b * (alpha.item() / a.shape[0])
        entries.extend((key, a, b_part) for key, b_part in _larry_targets(name, b, inner_dim))
    return {"label": label, "scale": 1.0, "entries": entries}


def _load_facial(inner_dim: int) -> dict:
    return _load_auto(FACIAL_REPO, FACIAL_FILE, inner_dim)


def _load_anime(inner_dim: int) -> dict:
    return _load_auto(ANIME_REPO, ANIME_FILE, inner_dim)


def _apply(entries, params, sign: float) -> None:
    for key, a, b in entries:
        param = params.get(key)
        if param is None:
            raise KeyError(f"LoRA target `{key}` not found in the transformer")
        delta = sign * (b.to(torch.float32) @ a.to(torch.float32))
        param.data = (param.data.float() + delta.to(param.device)).to(param.dtype)


def available() -> list[str]:
    """The LoRA sets that were loaded at startup, plus `off`."""
    state = getattr(_PIPE_TRANSFORMER, "_lora_state", None) if _PIPE_TRANSFORMER is not None else None
    return sorted(state["sets"]) + ["off"] if state else ["off"]


_PIPE_TRANSFORMER = None


def apply_lora(transformer) -> str | None:
    """Load every enabled LoRA set, fold the default one into `transformer`, and stash the factors for per-request
    switching. Returns a status line, or `None` when everything is disabled."""
    global _PIPE_TRANSFORMER
    _PIPE_TRANSFORMER = transformer

    inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    sets = {}
    if LARRY_FILE.lower() not in ("", "off", "none"):
        sets["larry"] = _load_larry(inner_dim)
    if os.environ.get("H3_LIGHTX", "on").lower() not in ("", "off", "none"):
        sets["lightx"] = _load_lightx()
    if os.environ.get("H3_LIGHTX8", "on").lower() not in ("", "off", "none"):
        sets["lightx8"] = _load_lightx(LIGHTX8_FILE)
    if os.environ.get("H3_REALISM", "on").lower() not in ("", "off", "none"):
        sets["realism"] = _load_realism(inner_dim)
    if os.environ.get("H3_JOYFOX", "on").lower() not in ("", "off", "none"):
        sets["joyfox"] = _load_joyfox(inner_dim)
    if os.environ.get("H3_FACIAL", "on").lower() not in ("", "off", "none"):
        sets[FACIAL_NAME] = _load_facial(inner_dim)
    if os.environ.get("H3_ANIME", "on").lower() not in ("", "off", "none"):
        sets[ANIME_NAME] = _load_anime(inner_dim)
    if not sets:
        return None

    active = DEFAULT_LORA if DEFAULT_LORA in sets else sorted(sets)[0]
    params = dict(transformer.named_parameters())
    _apply(sets[active]["entries"], params, sets[active]["scale"])
    transformer._lora_state = {"active": active, "sets": sets}
    return (
        f"LoRAs loaded: "
        + ", ".join(f"`{name}` ({spec['label']}, {len(spec['entries'])} weights)" for name, spec in sets.items())
        + f" · active `{active}`"
    )


def set_active(transformer, name: str) -> str:
    """Switch the folded LoRA in place. No-op when the state already matches. Returns the active set."""
    state = getattr(transformer, "_lora_state", None)
    if state is None:
        return "off"
    name = name if name in state["sets"] else "off"
    if state["active"] == name:
        return name
    params = dict(transformer.named_parameters())
    if state["active"] != "off":
        old = state["sets"][state["active"]]
        _apply(old["entries"], params, -old["scale"])
    if name != "off":
        _apply(state["sets"][name]["entries"], params, state["sets"][name]["scale"])
    state["active"] = name
    return name


def set_enabled(transformer, enabled: bool) -> bool:
    """Backwards-compatible boolean toggle over the default set."""
    return set_active(transformer, DEFAULT_LORA if enabled else "off") != "off"
