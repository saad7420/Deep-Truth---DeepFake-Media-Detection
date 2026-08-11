"""
M8 — Steganalysis Rich Model (SRM) filter bank and feature extraction.

WHAT THIS IS
────────────
A bank of fixed, zero-mean high-pass filters that suppress image *content*
and expose the local noise/prediction-error *residual*. Applied to a face
crop or a whole frame, the residual carries traces that survive most visual
edits: sensor noise consistency (or its absence after a splice), local
smoothing from denoising or upscaling, and boundary artefacts at the seam of
a composited region. This is a genuinely different signal from the ViViT/
ViT ensembles in M7 — those look at learned high-level content features;
this looks at low-level statistical texture that a generator or splice tool
rarely reproduces correctly, even when it fools the eye.

WHY 5 FILTERS, NOT THE LITERATURE'S 30
────────────────────────────────────────
The original "Spatial Rich Model" (Fridrich & Kodovsky, 2012) and its
adoption into deepfake/splicing detection (Zhou et al., "Learning Rich
Features for Image Manipulation Detection", CVPR 2018 — the paper this
module's design is descended from) both use a bank of ~30 fixed filters.
Zhou et al. themselves report that 3 of those 30 already capture nearly all
the benefit; the full 30 is a diminishing-returns tail, not a hard
requirement.

Reproducing those specific 30 (or even their specific 3) filter coefficients
from memory here carries a real risk this module does not take: a single
transposed sign or wrong normalisation constant in a "high-pass filter"
produces output that still *looks* plausible — small, textured, noise-like —
while being numerically wrong in a way nothing downstream would catch. That
is a worse failure than admitting the gap, so this bank uses only filters
whose correctness can be stated with certainty:

  KV      the second-order "Ker-Böhme" kernel — reproduced identically
          across enough independent steganalysis papers going back to 2004
          that transcription error is not a realistic concern
  HORIZ / VERT / DIAG / ADIAG
          first-order directional prediction-error filters, each simply
          "predict this pixel from its two neighbours along one direction,
          keep the error" — this is the literal, unambiguous definition
          Fridrich & Kodovsky build the whole rich-model family from
          (section III), not a memorised constant, so there is nothing to
          transcribe incorrectly

Every filter here is verified in test_srm.py: each has kernel-sum ≈ 0 (the
defining high-pass property — response to a constant region must be zero),
and each responds far more strongly to synthetic noise than to a flat
region. That is checked by running the actual convolution, not asserted.

If the literature's full filter bank is available later — e.g. sourced
directly from the CVPR 2018 authors' released code rather than reconstructed
from memory — `SRMFilterBank.from_npy()` below accepts any (N, k, k) array,
so extending this to 30 filters is a data change, not a code change.
"""
from __future__ import annotations

import numpy as np

# ── The verified 5-filter bank ────────────────────────────────────────────

_KV = np.array([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8,-12,  8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1],
], dtype=np.float32) / 12.0


def _directional(dy: int, dx: int) -> np.ndarray:
    """3-tap second-derivative residual along one direction, in a 5x5 frame
    (padded with zeros so every filter in the bank can share one conv call
    with a single kernel size)."""
    k = np.zeros((5, 5), dtype=np.float32)
    cy, cx = 2, 2
    k[cy, cx] = -2
    k[cy + dy, cx + dx] = 1
    k[cy - dy, cx - dx] = 1
    return k


FILTER_NAMES = ["kv", "horiz", "vert", "diag", "adiag"]

_DEFAULT_BANK = np.stack([
    _KV,
    _directional(0, 1),   # horiz
    _directional(1, 0),   # vert
    _directional(1, 1),   # diag
    _directional(1, -1),  # adiag
])  # shape (5, 5, 5)


class SRMFilterBank:
    """Holds a set of (k, k) high-pass kernels and applies them as one batched
    convolution, producing an (N, H', W') residual stack per input image."""

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
        """Load a (N, k, k) kernel array from disk — the extension point for
        dropping in a literature-sourced filter bank without touching code."""
        kernels = np.load(path)
        return cls(kernels, names)

    @property
    def n_filters(self) -> int:
        return self.kernels.shape[0]

    def apply(self, gray: np.ndarray) -> np.ndarray:
        """gray: (H, W) float32 in [0, 255]. Returns (N, H', W') residuals,
        one per filter, valid-mode (no border padding — SRM feature quality
        depends on genuine local statistics, not padded edge artefacts)."""
        from scipy.signal import convolve2d

        if gray.ndim != 2:
            raise ValueError(f"expected a single-channel (H, W) array, got {gray.shape}")

        return np.stack([
            convolve2d(gray, k, mode="valid")
            for k in self.kernels
        ]).astype(np.float32)


# ── Feature extraction: residuals → a fixed-length vector ──────────────────

# Truncation bound. SRM's own design quantises and clips residuals to a
# small range before pooling — an unbounded residual lets a handful of
# extreme pixels (a hot pixel, a compression block edge) dominate the whole
# feature, which defeats the purpose of a *statistical texture* descriptor.
TRUNCATION = 3.0


def extract_features(gray: np.ndarray, bank: SRMFilterBank | None = None) -> dict[str, float]:
    """Reduce one grayscale image to a fixed-length statistical feature dict.

    Per filter: truncated-residual mean, std, and the fraction of pixels
    that saturate the truncation bound (a proxy for how "hot"/structured the
    residual is — real sensor noise saturates rarely and evenly; splice
    boundaries and over-smoothed generator output saturate unevenly).

    This is deliberately a compact, interpretable statistic vector rather
    than a raw per-pixel co-occurrence matrix (the classical SRM feature,
    which runs into the tens of thousands of dimensions) — with only 5
    filters and a small classifier head downstream (see inferencers/srm.py),
    a dense feature would be mostly redundant and slower for no accuracy
    gain at this scale.
    """
    bank = bank or SRMFilterBank.default()
    residuals = bank.apply(gray)  # (N, H', W')

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
    """Same as extract_features but as an ordered float32 vector, for feeding
    a classifier. Order is fixed by FEATURE_NAMES so a trained checkpoint's
    input layout never depends on dict iteration order."""
    feats = extract_features(gray, bank)
    return np.array([feats[n] for n in feature_names(bank)], dtype=np.float32)


def feature_names(bank: SRMFilterBank | None = None) -> list[str]:
    bank = bank or SRMFilterBank.default()
    names = []
    for n in bank.names:
        names += [f"{n}_mean", f"{n}_std", f"{n}_sat_frac"]
    return names


# ── Patch-based pooling ──────────────────────────────────────────────────────
#
# extract_features()/feature_vector() above pool over the WHOLE image. That
# is the wrong tool for splice detection specifically: a splice is a LOCAL
# anomaly (one region's noise statistics differ from the rest), and
# averaging it together with the untouched majority of the frame washes the
# anomaly out almost completely.
#
# Measured directly (preprocessors/test_srm_filters.py::
# test_patch_pooling_beats_global_pooling_on_realistic_splice): with
# realistic between-photo variance — different scenes, different baseline
# noise levels, simulating different cameras — global pooling gives Cohen's
# d of roughly 0.15-0.25 between authentic and spliced images: a real but
# practically unlearnable signal, small enough that a classifier trained on
# it degrades to chance, which is what happened when this was tried against
# real CASIA v2 data (val EER ~48%, indistinguishable from random).
#
# The fix: compare each patch against the OTHER PATCHES IN THE SAME IMAGE,
# not against other photos. A spliced region's noise stands out from its own
# image's other regions even when it looks unremarkable next to a different
# photo's average — this is a within-image relative test instead of an
# across-image absolute one. Measured on the same synthetic setup: Cohen's d
# rises to ~1.2-1.3, a large, genuinely learnable effect size.

PATCH_SIZE = 32  # 224 / 32 = 7x7 = 49 patches — enough spatial resolution to
                 # localise a splice without each patch being too small for
                 # the 5x5 filters to produce a meaningful residual.


def _iter_patches(gray: np.ndarray, patch_size: int = PATCH_SIZE):
    """Yield fixed-size patches covering `gray`, with a few pixels of margin
    so the 5x5 valid-mode convolution still has enough border to work with."""
    h, w = gray.shape
    margin = 4
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            y0, y1 = max(0, y - margin), min(h, y + patch_size + margin)
            x0, x1 = max(0, x - margin), min(w, x + patch_size + margin)
            patch = gray[y0:y1, x0:x1]
            if patch.shape[0] >= 9 and patch.shape[1] >= 9:  # min size for 5x5 valid conv
                yield patch


def patch_feature_names(bank: SRMFilterBank | None = None) -> list[str]:
    base = feature_names(bank)
    return (
        [f"{n}_pmean" for n in base] +   # per-patch mean, ~= the old global feature
        [f"{n}_pstd" for n in base] +    # spread across patches — heterogeneity
        [f"{n}_poutlier" for n in base]  # the validated signal: how anomalous
                                          # is this image's most extreme patch,
                                          # relative to its OWN other patches
    )


def patch_feature_vector(gray: np.ndarray, bank: SRMFilterBank | None = None,
                         patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Split `gray` into a grid of patches, extract the base 15-dim stats
    per patch, and reduce to a fixed-length vector via three aggregations:
    mean, std, and a robust within-image outlier score. See the module
    docstring above this section for why the outlier score is the feature
    that actually catches a splice — the other two are kept because they
    are cheap and occasionally carry complementary signal, not because
    either alone would be sufficient.
    """
    bank = bank or SRMFilterBank.default()
    per_patch = np.stack([
        np.array([extract_features(p, bank)[n] for n in feature_names(bank)])
        for p in _iter_patches(gray, patch_size)
    ])  # (n_patches, 15)

    if per_patch.shape[0] < 2:
        # Image too small to tile — fall back to whole-image stats repeated
        # across all three aggregation slots rather than raising, since a
        # tiny/degenerate input is a data problem, not a code path that
        # should crash a batch job over one bad file.
        whole = feature_vector(gray, bank)
        return np.concatenate([whole, np.zeros_like(whole), np.zeros_like(whole)])

    p_mean = per_patch.mean(axis=0)
    p_std = per_patch.std(axis=0)

    median = np.median(per_patch, axis=0)
    mad = np.median(np.abs(per_patch - median), axis=0) + 1e-6  # robust spread,
    # not std — a single genuinely spliced patch should not inflate the
    # spread estimate used to judge how anomalous it is; MAD stays robust
    # to the one outlier it is trying to detect, ordinary std does not.
    p_outlier = np.abs(per_patch - median).max(axis=0) / mad

    return np.concatenate([p_mean, p_std, p_outlier]).astype(np.float32)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (H, W) float32. ITU-R BT.601 luma weights — the
    same convention OpenCV's cv2.COLOR_RGB2GRAY uses, so this is consistent
    with any grayscale conversion already happening elsewhere in the
    pipeline."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {rgb.shape}")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)