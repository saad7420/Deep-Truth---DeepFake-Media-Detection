"""Artifact maps — where in the image the ensemble found its evidence (M7 FE-3).

A score on its own is not evidence. "84% synthetic" tells an operator what the
model concluded, not what it saw, and there is no way to tell a real detection
from a confident mistake. This module turns each checkpoint's decision into a
spatial map of the regions that drove it.

Method
------
Grad-CAM, adapted for a vision transformer. The classic formulation assumes a
convolutional feature map with spatial axes; a ViT has none — after patch
embedding the image is a sequence of 196 tokens plus a CLS token. But those
196 tokens *are* a 14x14 grid, one token per 16x16 patch of a 224x224 input,
so the same arithmetic applies once the sequence is reshaped:

    1. forward, capturing the last block's pre-attention norm  (197, 768)
    2. backward from the "synthetic" logit                     (197, 768) grads
    3. drop CLS — it has no location, so it cannot be drawn
    4. channel weights = gradient mean over the 196 tokens     (768,)
    5. cam = ReLU(sum_c w_c * A_c) over channels               (196,)
    6. reshape to 14x14 and resize to the image

Which layer to hook is the one detail that decides whether this works at all,
and the obvious choice is wrong. `ViTForImageClassification` classifies from
the CLS token alone — `sequence_output[:, 0]`. So at the *output* of the last
encoder block, the 196 patch tokens feed nothing downstream, and
d(logit)/d(patch token) is exactly zero: hooking there yields an all-zero map,
silently, on every image.

The hook therefore goes on `layernorm_before` of the last block — the input
side, ahead of that block's attention. Patch tokens there still reach CLS
through the attention that follows, so gradients are non-zero and localised,
while the representation is as late (and as semantic) as it can be.

ReLU at step 5 is not cosmetic. Negative contributions are evidence *against*
the class, and leaving them in produces a map that lights up wherever the model
looked at all, rather than where it found something.

Why gradients rather than raw attention
---------------------------------------
Attention rollout is cheaper — no backward pass — but it shows where the model
*looked*, not what changed its mind, and it is class-agnostic: the same map
comes back whether the verdict was synthetic or authentic. Gradients are taken
with respect to a specific logit, so the map answers the question actually
being asked. The cost is one backward pass per checkpoint.

Honest limits
-------------
Grad-CAM is a saliency approximation, not a segmentation. At 14x14 the map is
coarse: it localises to roughly a sixteenth of the frame per cell, which is
enough to say "the mouth region" and not enough to trace a splice boundary.
And it explains the model, not the image — a checkpoint keying on a JPEG
artefact will produce a confident map over that artefact. `validate.py`'s
deletion test is what keeps this honest; a map that fails it is worse than no
map at all, because it looks like evidence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: ViT-B/16 at 224x224 → 14x14 patch grid.
GRID = 14

#: Class index the maps explain. Matches the training-time convention used by
#: every image_*_lora_best checkpoint (0 = real, 1 = fake).
FAKE_CLASS = 1


# ─────────────────────────────────────────────────────────────────────────────
# CAM computation
# ─────────────────────────────────────────────────────────────────────────────

#: Attribute the transformer trunk hides under, per architecture. ViT for
#: images, ViViT for video — the two are structurally identical from this
#: module's point of view, differing only in how many tokens come back.
_TRUNK_ATTRS = ("vit", "vivit")


def _cam_target_layer(call_target):
    """The module to hook: the last block's pre-attention layernorm.

    See the module docstring for why this is not the last block's *output* —
    patch tokens there have zero gradient, because the classifier reads CLS
    only. That is true of `VivitForVideoClassification` for exactly the same
    reason it is true of the image model, so both take this path. Falls back
    to the second-to-last block's output if the attribute is ever renamed
    upstream.

    The trunk lookup is defensive: the model may arrive bare or wrapped by
    PEFT, and the two do not agree on where the submodule lives.
    """
    trunk = None
    for holder in (call_target, getattr(call_target, "base_model", None)):
        if holder is None:
            continue
        for attr in _TRUNK_ATTRS:
            trunk = getattr(holder, attr, None)
            if trunk is not None:
                break
        if trunk is not None:
            break

    if trunk is None:
        raise AttributeError("could not locate the transformer trunk for hooking")

    layers = trunk.encoder.layer
    target = getattr(layers[-1], "layernorm_before", None)
    if target is not None:
        return target
    if len(layers) >= 2:
        log.debug("artifact map: layernorm_before missing, hooking layer[-2]")
        return layers[-2]
    raise AttributeError("no usable layer to hook for Grad-CAM")


def _token_relevance(model, pixel_values, target_class: int,
                     forward_kwargs: dict | None = None) -> np.ndarray | None:
    """Per-patch-token relevance, CLS dropped. Shared by image and video.

    Returns a 1-D array of length n_patch_tokens, un-normalised and already
    passed through ReLU, or None if no map could be produced. The caller
    reshapes it: 14x14 for a still, (T, 14, 14) for a clip.
    """
    import torch

    # Same PEFT forward-signature workaround as the inferencers use.
    call_target = model.get_base_model() if hasattr(model, "get_base_model") else model

    try:
        block = _cam_target_layer(call_target)
    except AttributeError as exc:
        log.warning(f"artifact map: {exc}")
        return None

    captured: dict[str, Any] = {}

    def _hook(_module, _inputs, output):
        act = output[0] if isinstance(output, tuple) else output
        act.retain_grad()
        captured["act"] = act

    handle = block.register_forward_hook(_hook)

    try:
        # The graph only exists if something upstream requires grad. Every
        # parameter is frozen at inference (eval mode, LoRA adapters loaded
        # read-only), so without this the backward pass has nothing to
        # traverse and `.grad` comes back None.
        px = pixel_values.detach().clone().requires_grad_(True)

        # Deliberately outside any autocast block. Half precision makes the
        # backward pass numerically unstable enough to produce all-zero maps
        # on the checkpoints that score lowest, which is exactly where a map
        # would be most misleading.
        with torch.enable_grad():
            out = call_target(pixel_values=px, **(forward_kwargs or {}))
            logit = out.logits[0, target_class]
            call_target.zero_grad(set_to_none=True)
            logit.backward()

        act = captured.get("act")
        if act is None or act.grad is None:
            log.warning("artifact map: no gradient reached the hooked block")
            return None

        # (1, 1 + n_patches, 768) -> drop CLS, which has no position to draw.
        a = act.detach()[0, 1:, :]
        g = act.grad.detach()[0, 1:, :]

        weights = g.mean(dim=0)                       # (768,)
        rel = torch.relu((a * weights).sum(dim=-1))   # (n_patches,)
        return rel.cpu().numpy().astype(np.float32)

    except Exception as exc:  # noqa: BLE001 — a map is never worth a failed case
        log.warning(f"artifact map: CAM computation failed: {exc}")
        return None
    finally:
        handle.remove()


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Scale to [0,1], or return all-zeros when there is nothing to scale.

    A flat-zero map means the checkpoint found nothing it considers
    synthetic. That is the truthful output; normalising it would manufacture
    structure out of noise.
    """
    peak = float(arr.max()) if arr.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / peak).astype(np.float32)


def compute_cam(model, pixel_values, target_class: int = FAKE_CLASS) -> np.ndarray | None:
    """A 14x14 relevance map for a still image, normalised to [0, 1].

    Returns None if anything prevents a map being produced. Callers treat that
    as "no map for this checkpoint", never as an error: a missing overlay is a
    cosmetic loss, a failed case is not.
    """
    rel = _token_relevance(model, pixel_values, target_class)
    if rel is None:
        return None

    if rel.size != GRID * GRID:
        log.warning(f"artifact map: expected {GRID * GRID} patch tokens, "
                    f"got {rel.size}")
        return None

    return _normalise(rel.reshape(GRID, GRID))


def compute_video_cam(model, pixel_values,
                      target_class: int = FAKE_CLASS) -> np.ndarray | None:
    """A (T, 14, 14) relevance cube for a clip, normalised to [0, 1] overall.

    ViViT embeds *tubelets* — a 2x16x16 block of two frames by sixteen pixels
    square — so its 1568 tokens are 8 temporal segments of a 14x14 spatial
    grid, not one flat grid. Reshaping to (T, 14, 14) recovers that structure,
    which is what makes a video map more informative than an image one: it
    localises in time as well as space, so the report can say which moment of
    the clip carried the evidence.

    Normalised across the whole cube rather than per segment. Per-segment
    normalisation would rescale every segment to peak at 1.0, erasing exactly
    the difference the temporal profile exists to show — a quiet segment would
    look as incriminating as the decisive one.

    `interpolate_pos_encoding=True` mirrors VideoInferencer._predict_one: the
    checkpoints run 16 frames against a backbone pretrained on 32, and the
    positional embeddings are resized to match.
    """
    rel = _token_relevance(model, pixel_values, target_class,
                           forward_kwargs={"interpolate_pos_encoding": True})
    if rel is None:
        return None

    per_frame = GRID * GRID
    if rel.size % per_frame != 0 or rel.size == 0:
        log.warning(f"artifact map: {rel.size} tokens is not a whole number of "
                    f"{GRID}x{GRID} segments")
        return None

    segments = rel.size // per_frame
    return _normalise(rel.reshape(segments, GRID, GRID))


def place_in_frame(cam: np.ndarray, frame_h: int, frame_w: int,
                   rect: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Lift a 14x14 CAM into full-frame coordinates.

    `rect` is (x1, y1, x2, y2) in frame pixels, for a checkpoint that saw a
    crop rather than the whole image. Without it the map is simply stretched
    over the frame.

    This exists because the face-crop checkpoint's map is expressed in *crop*
    space. Compositing it directly onto the whole image would draw the mouth
    over somebody's shoulder — a forensic tool pointing at the wrong region is
    worse than one that points nowhere, so the geometry is undone explicitly
    and everything is combined in one coordinate system.
    """
    if rect is None:
        return _resize_nearest_smooth(cam, frame_h, frame_w)

    x1, y1, x2, y2 = rect
    x1 = max(0, min(frame_w - 1, int(x1)))
    y1 = max(0, min(frame_h - 1, int(y1)))
    x2 = max(x1 + 1, min(frame_w, int(x2)))
    y2 = max(y1 + 1, min(frame_h, int(y2)))

    placed = np.zeros((frame_h, frame_w), dtype=np.float32)
    placed[y1:y2, x1:x2] = _resize_nearest_smooth(cam, y2 - y1, x2 - x1)
    return placed


def combine(frames: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray | None:
    """Fuse per-checkpoint frame-space maps into one.

    Weighted by P(fake) rather than uniformly, because a checkpoint reporting
    1% synthetic is not asserting that anything is wrong anywhere — averaging
    its map in would dilute the localisation of the checkpoints that are
    actually making a claim. This mirrors what the ensemble does with the
    scores themselves: contributions count in proportion to their strength.
    """
    usable = {k: v for k, v in frames.items() if v is not None and v.max() > 0}
    if not usable:
        return None

    shape = next(iter(usable.values())).shape
    total = sum(max(0.0, weights.get(k, 0.0)) for k in usable)

    if total <= 0:
        # Nothing claimed manipulation; fall back to an unweighted mean so the
        # map still shows where the models were looking.
        combined = np.stack(list(usable.values())).mean(axis=0)
    else:
        combined = np.zeros(shape, dtype=np.float32)
        for slug, frame in usable.items():
            combined += frame * (max(0.0, weights.get(slug, 0.0)) / total)

    peak = float(combined.max())
    return combined / peak if peak > 0 else combined


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _colourise(cam: np.ndarray) -> np.ndarray:
    """Map [0,1] relevance to RGB with a blue → cyan → yellow → red ramp.

    Hand-rolled rather than pulled from matplotlib: this is the only thing the
    pipeline would need that dependency for, and a four-stop piecewise ramp is
    a few lines. The ramp is monotonic in luminance as well as hue so the map
    still reads when printed in greyscale — forensic reports get printed.
    """
    x = np.clip(cam, 0.0, 1.0)
    stops = np.array([
        [0.03, 0.05, 0.28],   # 0.00 near-black blue
        [0.10, 0.55, 0.75],   # 0.33 cyan
        [0.95, 0.85, 0.25],   # 0.66 yellow
        [0.85, 0.10, 0.10],   # 1.00 red
    ], dtype=np.float32)

    pos = x * (len(stops) - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    frac = (pos - lo)[..., None]
    rgb = stops[lo] * (1.0 - frac) + stops[lo + 1] * frac
    return (rgb * 255.0).astype(np.uint8)


def _resize_nearest_smooth(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    """Upsample the 14x14 map with bilinear interpolation via PIL.

    Bilinear, not nearest: a blocky 14x14 grid implies the model localised to
    exact patch boundaries, which it did not. Smooth gradients read as the
    approximation this is.
    """
    from PIL import Image
    img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize((width, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def render_overlay(base_rgb: np.ndarray, cam: np.ndarray, out_path: str,
                   alpha: float = 0.65, floor: float = 0.15) -> str:
    """Write `base_rgb` with the heat map composited over it.

    `floor` keeps low-relevance areas nearly untouched. Blending the whole
    frame uniformly tints regions the model had no opinion about, which reads
    as "the system suspects everything" — the opposite of localisation.
    """
    from PIL import Image

    h, w = base_rgb.shape[:2]
    up = _resize_nearest_smooth(cam, h, w)
    heat = _colourise(up).astype(np.float32)

    # Per-pixel opacity, so the overlay fades out where there is nothing to say.
    strength = np.clip((up - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
    a = (strength * alpha)[..., None]

    blended = base_rgb.astype(np.float32) * (1.0 - a) + heat * a
    Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(out_path)
    return out_path


def render_video_overlay(frames: np.ndarray, cube: np.ndarray, out_path: str,
                         columns: int = 4, alpha: float = 0.65,
                         floor: float = 0.15, label_band: int = 18) -> str:
    """Contact sheet: each temporal segment's heat over its own frame.

    A single blended image cannot represent a clip — averaging the cube over
    time would hide the one segment that mattered, which for a face swap is
    frequently a handful of frames. So each of the T segments is drawn
    separately, over the frame it actually covers, and laid out in a grid that
    reads left-to-right in time.

    Each segment spans `tubelet` frames (2 for this backbone); the first is
    used as its representative. A thin band under each tile carries the
    segment's share of total relevance, so the strongest moment is findable
    without comparing colours by eye.
    """
    from PIL import Image, ImageDraw

    segments = cube.shape[0]
    if segments == 0 or frames.size == 0:
        raise ValueError("nothing to render")

    fh, fw = frames.shape[1:3]
    per_segment = max(1, len(frames) // segments)
    profile = temporal_profile(cube)

    columns = max(1, min(columns, segments))
    rows = (segments + columns - 1) // columns
    tile_h = fh + label_band

    sheet = Image.new("RGB", (columns * fw, rows * tile_h), (8, 12, 20))
    draw = ImageDraw.Draw(sheet)

    for t in range(segments):
        frame = frames[min(t * per_segment, len(frames) - 1)]
        heat = place_in_frame(cube[t], fh, fw)

        tmp = f"{out_path}.seg{t}.tmp.png"
        render_overlay(frame, heat, tmp, alpha=alpha, floor=floor)
        with Image.open(tmp) as tile:
            tile.load()
            x, y = (t % columns) * fw, (t // columns) * tile_h
            sheet.paste(tile, (x, y))
        Path(tmp).unlink(missing_ok=True)

        share = profile[t] if t < len(profile) else 0.0
        draw.text((x + 4, y + fh + 4),
                  f"seg {t + 1}/{segments}   {share * 100:.0f}%",
                  fill=(148, 163, 184))
        # Bar the width of this segment's share, so the eye finds the peak
        # without reading the numbers.
        draw.rectangle([x + 4, y + fh + label_band - 4,
                        x + 4 + int((fw - 8) * share), y + fh + label_band - 2],
                       fill=(239, 68, 68))

    sheet.save(out_path)
    return out_path


def temporal_profile(cube: np.ndarray) -> list[float]:
    """Each segment's share of the clip's total relevance, summing to 1.

    Intended to answer *when* the model found something: a spike would mean a
    few frames carry the evidence (a swap, a splice, one edited shot), a flat
    profile that it is spread across the clip.

    Measured caveat, and it is a significant one. On the clips tested so far
    the profile comes back very close to uniform — 0.108 to 0.128 against an
    even share of 0.125 — and a per-segment occlusion test showed the measured
    impact of removing each segment is equally flat. So this has *not* been
    demonstrated to discriminate in time. The likely cause is the channel
    weighting: gradients are averaged over all 1568 tokens, across time as
    well as space, which washes out temporal contrast before the profile is
    computed.

    It is kept because the number is honest as a description of the cube, and
    because `summarise_video` gates every temporal *claim* behind
    `temporally_localised`, which on this evidence correctly returns False.
    Treat a True from it as untested rather than proven until a clip with a
    known localised edit says otherwise. The spatial half of the map is on
    much firmer ground — see the deletion test in the README.
    """
    per_segment = cube.reshape(cube.shape[0], -1).sum(axis=1)
    total = float(per_segment.sum())
    if total <= 0:
        return [0.0] * cube.shape[0]
    return [round(float(v / total), 4) for v in per_segment]


def summarise_video(cube: np.ndarray, top_k: int = 3) -> dict:
    """Region summary for a clip, plus where in time the evidence sits."""
    profile = temporal_profile(cube)
    collapsed = cube.max(axis=0)          # strongest moment per location
    record = summarise(_normalise(collapsed), top_k=top_k)

    peak = int(np.argmax(profile)) if profile else 0
    record.update({
        "segments": int(cube.shape[0]),
        "temporal_profile": profile,
        "peak_segment": peak,
        # Even spread over T segments is 1/T each. A peak carrying more than
        # twice its even share is a moment worth pointing at; anything less is
        # a clip whose evidence is genuinely spread out, and saying otherwise
        # would invent a timestamp.
        "temporally_localised": bool(
            profile and cube.shape[0] > 1
            and profile[peak] > 2.0 / cube.shape[0]
        ),
    })
    return record


def summarise(frame_map: np.ndarray, top_k: int = 3) -> dict:
    """Machine-readable description of where the map concentrates.

    The overlay is for a human; this is for the report row, the API, and
    anything that needs to reason about the map without decoding a PNG.
    Coordinates are fractions of the frame so they survive any resize.

    Reported on a GRID x GRID summary of the full-resolution map rather than
    per pixel: the underlying CAM has 14x14 of real spatial resolution, and
    quoting regions any finer would claim precision the method does not have.
    """
    from PIL import Image
    small = Image.fromarray(
        (np.clip(frame_map, 0, 1) * 255).astype(np.uint8), mode="L"
    ).resize((GRID, GRID), Image.BILINEAR)
    cam = np.asarray(small, dtype=np.float32) / 255.0

    flat = cam.reshape(-1)
    order = np.argsort(flat)[::-1][:top_k]

    regions = []
    for idx in order:
        if flat[idx] <= 0:
            continue
        row, col = divmod(int(idx), GRID)
        regions.append({
            "x": round(col / GRID, 4),
            "y": round(row / GRID, 4),
            "w": round(1.0 / GRID, 4),
            "h": round(1.0 / GRID, 4),
            "relevance": round(float(flat[idx]), 4),
        })

    # Share of total relevance held by the strongest 10% of cells. High values
    # mean a genuinely localised finding; near-uniform means the model is
    # responding to the image as a whole (global texture, compression), which
    # is a real answer but not a *located* one.
    cells = max(1, int(round(0.1 * flat.size)))
    top_mass = float(np.sort(flat)[::-1][:cells].sum())
    total = float(flat.sum())

    return {
        "grid": GRID,
        "regions": regions,
        "concentration": round(top_mass / total, 4) if total > 0 else 0.0,
        "localised": bool(total > 0 and (top_mass / total) > 0.25),
    }
