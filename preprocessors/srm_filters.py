from __future__ import annotations

import numpy as np


_KV = np.array([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8,-12,  8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1],
], dtype=np.float32) / 12.0


def _directional(dy: int, dx: int) -> np.ndarray:
    k = np.zeros((5, 5), dtype=np.float32)
    cy, cx = 2, 2
    k[cy, cx] = -2
    k[cy + dy, cx + dx] = 1
    k[cy - dy, cx - dx] = 1
    return k


FILTER_NAMES = ["kv", "horiz", "vert", "diag", "adiag"]

_DEFAULT_BANK = np.stack([
    _KV,
    _directional(0, 1),
    _directional(1, 0),
    _directional(1, 1),
    _directional(1, -1),
])


# five high-pass filters that keep only the noise residual
class SRMFilterBank:

    def __init__(self, kernels: np.ndarray, names: list[str] | None = None):
        if kernels.ndim != 3 or kernels.shape[1] != kernels.shape[2]:
            raise ValueError(
                f"expected kernels shaped (N, k, k), got {kernels.shape}")
        self.kernels = kernels.astype(np.float32)
        self.names = names or [f"f{i}" for i in range(kernels.shape[0])]
        if len(self.names) != kernels.shape[0]:
            raise ValueError("names length must match kernel count")

    @classmethod
    def default(cls) -> "SRMFilterBank":
        return cls(_DEFAULT_BANK, FILTER_NAMES)

    @classmethod
    def from_npy(cls, path: str, names: list[str] | None = None) -> "SRMFilterBank":
        kernels = np.load(path)
        return cls(kernels, names)

    @property
    def n_filters(self) -> int:
        return self.kernels.shape[0]

    def apply(self, gray: np.ndarray) -> np.ndarray:
        from scipy.signal import convolve2d

        if gray.ndim != 2:
            raise ValueError(f"expected a single-channel (H, W) array, got {gray.shape}")

        return np.stack([
            convolve2d(gray, k, mode="valid")
            for k in self.kernels
        ]).astype(np.float32)


TRUNCATION = 3.0


def extract_features(gray: np.ndarray, bank: SRMFilterBank | None = None) -> dict[str, float]:
    bank = bank or SRMFilterBank.default()
    residuals = bank.apply(gray)

    truncated = np.clip(residuals, -TRUNCATION, TRUNCATION)

    feats: dict[str, float] = {}
    for i, name in enumerate(bank.names):
        r = truncated[i]
        feats[f"{name}_mean"] = float(r.mean())
        feats[f"{name}_std"] = float(r.std())
        feats[f"{name}_sat_frac"] = float(
            (np.abs(residuals[i]) >= TRUNCATION).mean())

    return feats


def feature_vector(gray: np.ndarray, bank: SRMFilterBank | None = None) -> np.ndarray:
    feats = extract_features(gray, bank)
    return np.array([feats[n] for n in feature_names(bank)], dtype=np.float32)


def feature_names(bank: SRMFilterBank | None = None) -> list[str]:
    bank = bank or SRMFilterBank.default()
    names = []
    for n in bank.names:
        names += [f"{n}_mean", f"{n}_std", f"{n}_sat_frac"]
    return names


PATCH_SIZE = 32


def _iter_patches(gray: np.ndarray, patch_size: int = PATCH_SIZE):
    h, w = gray.shape
    margin = 4
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            y0, y1 = max(0, y - margin), min(h, y + patch_size + margin)
            x0, x1 = max(0, x - margin), min(w, x + patch_size + margin)
            patch = gray[y0:y1, x0:x1]
            if patch.shape[0] >= 9 and patch.shape[1] >= 9:
                yield patch


def patch_feature_names(bank: SRMFilterBank | None = None) -> list[str]:
    base = feature_names(bank)
    return (
        [f"{n}_pmean" for n in base] +
        [f"{n}_pstd" for n in base] +
        [f"{n}_poutlier" for n in base]
    )


# patch pooling lets a spliced region stand out from the rest
def patch_feature_vector(gray: np.ndarray, bank: SRMFilterBank | None = None,
                         patch_size: int = PATCH_SIZE) -> np.ndarray:
    bank = bank or SRMFilterBank.default()
    per_patch = np.stack([
        np.array([extract_features(p, bank)[n] for n in feature_names(bank)])
        for p in _iter_patches(gray, patch_size)
    ])

    if per_patch.shape[0] < 2:
        whole = feature_vector(gray, bank)
        return np.concatenate([whole, np.zeros_like(whole), np.zeros_like(whole)])

    p_mean = per_patch.mean(axis=0)
    p_std = per_patch.std(axis=0)

    median = np.median(per_patch, axis=0)
    mad = np.median(np.abs(per_patch - median), axis=0) + 1e-6
    p_outlier = np.abs(per_patch - median).max(axis=0) / mad

    return np.concatenate([p_mean, p_std, p_outlier]).astype(np.float32)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {rgb.shape}")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
