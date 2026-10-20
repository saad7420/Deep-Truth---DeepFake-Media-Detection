from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

GRID = 14

FAKE_CLASS = 1


_TRUNK_ATTRS = ("vit", "vivit")


# hook before the last block, after it the patch gradients are zero
def _cam_target_layer(call_target):
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
    import torch

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
        px = pixel_values.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            out = call_target(pixel_values=px, **(forward_kwargs or {}))
            logit = out.logits[0, target_class]
            call_target.zero_grad(set_to_none=True)
            logit.backward()

        act = captured.get("act")
        if act is None or act.grad is None:
            log.warning("artifact map: no gradient reached the hooked block")
            return None

        a = act.detach()[0, 1:, :]
        g = act.grad.detach()[0, 1:, :]

        weights = g.mean(dim=0)
        rel = torch.relu((a * weights).sum(dim=-1))
        return rel.cpu().numpy().astype(np.float32)

    except Exception as exc:
        log.warning(f"artifact map: CAM computation failed: {exc}")
        return None
    finally:
        handle.remove()


def _normalise(arr: np.ndarray) -> np.ndarray:
    peak = float(arr.max()) if arr.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / peak).astype(np.float32)


def compute_cam(model, pixel_values, target_class: int = FAKE_CLASS) -> np.ndarray | None:
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


# weight each checkpoint's map by how strongly it flagged the input
def combine(frames: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray | None:
    usable = {k: v for k, v in frames.items() if v is not None and v.max() > 0}
    if not usable:
        return None

    shape = next(iter(usable.values())).shape
    total = sum(max(0.0, weights.get(k, 0.0)) for k in usable)

    if total <= 0:
        combined = np.stack(list(usable.values())).mean(axis=0)
    else:
        combined = np.zeros(shape, dtype=np.float32)
        for slug, frame in usable.items():
            combined += frame * (max(0.0, weights.get(slug, 0.0)) / total)

    peak = float(combined.max())
    return combined / peak if peak > 0 else combined


def _colourise(cam: np.ndarray) -> np.ndarray:
    x = np.clip(cam, 0.0, 1.0)
    stops = np.array([
        [0.03, 0.05, 0.28],
        [0.10, 0.55, 0.75],
        [0.95, 0.85, 0.25],
        [0.85, 0.10, 0.10],
    ], dtype=np.float32)

    pos = x * (len(stops) - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    frac = (pos - lo)[..., None]
    rgb = stops[lo] * (1.0 - frac) + stops[lo + 1] * frac
    return (rgb * 255.0).astype(np.uint8)


def _resize_nearest_smooth(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize((width, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def render_overlay(base_rgb: np.ndarray, cam: np.ndarray, out_path: str,
                   alpha: float = 0.65, floor: float = 0.15) -> str:
    from PIL import Image

    h, w = base_rgb.shape[:2]
    up = _resize_nearest_smooth(cam, h, w)
    heat = _colourise(up).astype(np.float32)

    strength = np.clip((up - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
    a = (strength * alpha)[..., None]

    blended = base_rgb.astype(np.float32) * (1.0 - a) + heat * a
    Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(out_path)
    return out_path


def render_video_overlay(frames: np.ndarray, cube: np.ndarray, out_path: str,
                         columns: int = 4, alpha: float = 0.65,
                         floor: float = 0.15, label_band: int = 18) -> str:
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
        draw.rectangle([x + 4, y + fh + label_band - 4,
                        x + 4 + int((fw - 8) * share), y + fh + label_band - 2],
                       fill=(239, 68, 68))

    sheet.save(out_path)
    return out_path


def temporal_profile(cube: np.ndarray) -> list[float]:
    per_segment = cube.reshape(cube.shape[0], -1).sum(axis=1)
    total = float(per_segment.sum())
    if total <= 0:
        return [0.0] * cube.shape[0]
    return [round(float(v / total), 4) for v in per_segment]


def summarise_video(cube: np.ndarray, top_k: int = 3) -> dict:
    profile = temporal_profile(cube)
    collapsed = cube.max(axis=0)
    record = summarise(_normalise(collapsed), top_k=top_k)

    peak = int(np.argmax(profile)) if profile else 0
    record.update({
        "segments": int(cube.shape[0]),
        "temporal_profile": profile,
        "peak_segment": peak,
        "temporally_localised": bool(
            profile and cube.shape[0] > 1
            and profile[peak] > 2.0 / cube.shape[0]
        ),
    })
    return record


def summarise(frame_map: np.ndarray, top_k: int = 3) -> dict:
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

    cells = max(1, int(round(0.1 * flat.size)))
    top_mass = float(np.sort(flat)[::-1][:cells].sum())
    total = float(flat.sum())

    return {
        "grid": GRID,
        "regions": regions,
        "concentration": round(top_mass / total, 4) if total > 0 else 0.0,
        "localised": bool(total > 0 and (top_mass / total) > 0.25),
    }
