"""
Numerical verification for preprocessors/srm_filters.py.

Run directly (no pytest dependency needed, matches this repo's other
throwaway-verification-script style):

    python -m preprocessors.test_srm_filters

Every assertion here checks an actual mathematical property by computing it,
not by inspection — these are the same checks that were run interactively
before this filter bank was committed to the pipeline.
"""
from __future__ import annotations

import numpy as np

from .srm_filters import (
    SRMFilterBank,
    extract_features,
    feature_names,
    feature_vector,
    patch_feature_names,
    patch_feature_vector,
    to_grayscale,
)


def test_filters_are_zero_mean():
    """Defining property of a high-pass/residual filter: a constant region
    must produce exactly zero response, or it isn't actually suppressing
    content."""
    bank = SRMFilterBank.default()
    const = np.full((32, 32), 128.0, dtype=np.float32)
    out = bank.apply(const)
    max_response = np.abs(out).max()
    assert max_response < 1e-3, (
        f"filter bank responded to a constant image (max={max_response}); "
        f"a non-zero-sum kernel slipped in")
    print(f"  zero-mean check: max response on constant input = {max_response:.2e}  PASS")


def test_filters_discriminate_noise():
    """The entire justification for using these as a forensic signal: they
    must respond far more to local statistical texture than to smoothness."""
    rng = np.random.default_rng(0)
    bank = SRMFilterBank.default()
    smooth = np.full((64, 64), 128.0, dtype=np.float32)
    noisy = (128 + rng.normal(0, 15, (64, 64))).astype(np.float32)

    e_smooth = np.abs(bank.apply(smooth)).mean()
    e_noisy = np.abs(bank.apply(noisy)).mean()
    ratio = e_noisy / max(e_smooth, 1e-9)

    assert ratio > 100, f"noise discrimination too weak: ratio={ratio:.1f}x"
    print(f"  discrimination check: noisy/smooth residual energy ratio = {ratio:,.0f}x  PASS")


def test_feature_vector_is_deterministic_and_ordered():
    """A checkpoint trained against this feature layout must get the exact
    same layout back at inference time, every time — this is the checkpoint
    contract equivalent of what UNFREEZE_LAYERS/HEAD_DIMS are for the audio
    module."""
    rng = np.random.default_rng(1)
    gray = rng.uniform(0, 255, (64, 64)).astype(np.float32)

    v1 = feature_vector(gray)
    v2 = feature_vector(gray)
    assert np.array_equal(v1, v2), "feature_vector is not deterministic"

    names = feature_names()
    assert len(names) == len(v1), (
        f"feature_names() length ({len(names)}) != feature_vector() length "
        f"({len(v1)}) — a classifier head sized against one would silently "
        f"misalign against the other")

    bank = SRMFilterBank.default()
    expected_len = bank.n_filters * 3  # mean, std, sat_frac per filter
    assert len(v1) == expected_len, (
        f"expected {expected_len} features ({bank.n_filters} filters x 3 "
        f"stats), got {len(v1)}")

    print(f"  feature vector: {len(v1)}-dim, deterministic, "
          f"names aligned  PASS")


def test_extract_features_matches_vector():
    """The dict and array forms must describe the same numbers in the same
    order — extract_features() is for readable evidence/debugging,
    feature_vector() is what a classifier consumes; they must never drift
    apart."""
    rng = np.random.default_rng(2)
    gray = rng.uniform(0, 255, (48, 48)).astype(np.float32)

    feats = extract_features(gray)
    vec = feature_vector(gray)
    names = feature_names()

    for i, name in enumerate(names):
        assert abs(feats[name] - vec[i]) < 1e-5, (
            f"mismatch at {name}: dict={feats[name]} vector={vec[i]}")
    print(f"  dict/vector consistency: all {len(names)} values match  PASS")


def test_to_grayscale_shape_and_range():
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (32, 32, 3)).astype(np.uint8)
    gray = to_grayscale(rgb)
    assert gray.shape == (32, 32)
    assert gray.dtype == np.float32
    assert 0 <= gray.min() and gray.max() <= 255
    print(f"  to_grayscale: shape={gray.shape} dtype={gray.dtype} "
          f"range=[{gray.min():.1f}, {gray.max():.1f}]  PASS")


def test_real_vs_synthetic_edge_case():
    """A sharp, hard-edged rectangle (the kind a crude splice boundary
    produces) must register a much stronger residual than natural smooth
    gradients — this is the actual forensic mechanism, checked directly
    rather than assumed."""
    bank = SRMFilterBank.default()

    natural = np.zeros((64, 64), dtype=np.float32)
    for i in range(64):
        natural[i, :] = 100 + i * 0.5  # smooth gradient, no hard edges

    spliced = natural.copy()
    spliced[20:44, 20:44] = 200.0  # hard-edged inserted block

    e_natural = np.abs(bank.apply(natural)).mean()
    e_spliced = np.abs(bank.apply(spliced)).mean()

    assert e_spliced > e_natural * 2, (
        f"hard-edged region did not register as more anomalous: "
        f"natural={e_natural:.3f} spliced={e_spliced:.3f}")
    print(f"  splice-edge check: natural={e_natural:.3f}  "
          f"spliced={e_spliced:.3f}  ({e_spliced / e_natural:.1f}x)  PASS")


def test_patch_pooling_beats_global_pooling_on_realistic_splice():
    """The test that actually justifies patch_feature_vector()'s existence.

    A too-easy synthetic splice (a huge, high-contrast hard-edged block)
    would pass with either global or patch pooling and prove nothing. This
    constructs a genuinely hard case instead: a small (60x60 in 224x224,
    ~7% of the frame) splice with realistic between-photo variance — every
    sample uses a different scene gradient and a different baseline noise
    level, simulating different real cameras — which is what training
    against real CASIA v2 data actually looks like.

    Global pooling measurably fails this case (this is not hypothetical —
    running real CASIA v2 through the whole-image feature_vector() and
    training a classifier on it gave val EER ~48%, indistinguishable from
    chance). Patch-based pooling, comparing each patch against the other
    patches in its OWN image rather than against other photos, must show a
    substantially larger effect size on the same synthetic setup, or this
    redesign has no evidence behind it.
    """
    from .srm_filters import feature_vector, patch_feature_vector, feature_names

    rng = np.random.default_rng(42)

    def make_scene(splice: bool):
        slope = rng.uniform(-0.5, 0.5)
        base_std = rng.uniform(4, 14)
        base = np.zeros((224, 224), dtype=np.float32)
        for i in range(224):
            base[i, :] = 100 + i * slope
        img = base + rng.normal(0, base_std, (224, 224))
        if splice:
            y0, x0 = rng.integers(20, 140), rng.integers(20, 140)
            splice_std = base_std * rng.uniform(1.8, 3.0)
            img[y0:y0 + 60, x0:x0 + 60] = (
                base[y0:y0 + 60, x0:x0 + 60] + rng.normal(0, splice_std, (60, 60)))
        return np.clip(img, 0, 255).astype(np.float32)

    def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
        return abs(a.mean() - b.mean()) / max(pooled, 1e-9)

    n = 40
    authentic = [make_scene(False) for _ in range(n)]
    spliced = [make_scene(True) for _ in range(n)]

    # Global pooling: worst-case single feature, matching what training
    # actually saw.
    names = feature_names()
    kv_std_idx = names.index("kv_std")
    global_a = np.array([feature_vector(im)[kv_std_idx] for im in authentic])
    global_s = np.array([feature_vector(im)[kv_std_idx] for im in spliced])
    d_global = cohens_d(global_a, global_s)

    # Patch pooling: the outlier feature for the same underlying statistic.
    patch_names = patch_feature_names()
    outlier_idx = patch_names.index("kv_std_poutlier")
    patch_a = np.array([patch_feature_vector(im)[outlier_idx] for im in authentic])
    patch_s = np.array([patch_feature_vector(im)[outlier_idx] for im in spliced])
    d_patch = cohens_d(patch_a, patch_s)

    print(f"  global pooling (kv_std):          Cohen's d = {d_global:.2f}")
    print(f"  patch pooling (kv_std_poutlier):   Cohen's d = {d_patch:.2f}")

    assert d_global < 0.5, (
        f"global pooling was expected to struggle on this realistic case "
        f"(d={d_global:.2f}) — if it no longer does, the synthetic setup "
        f"has drifted from what actually failed against real CASIA data "
        f"and this test needs revisiting")
    assert d_patch > d_global * 2, (
        f"patch pooling ({d_patch:.2f}) should substantially beat global "
        f"pooling ({d_global:.2f}) on a local anomaly — it didn't")
    assert d_patch > 0.8, (
        f"patch pooling's effect size ({d_patch:.2f}) is not large enough "
        f"to call this validated — a real classifier needs more than this "
        f"to learn from")
    print(f"  PASS — patch pooling is "
          f"{d_patch / max(d_global, 1e-9):.1f}x more separable")


def run_all():
    tests = [
        test_filters_are_zero_mean,
        test_filters_discriminate_noise,
        test_feature_vector_is_deterministic_and_ordered,
        test_extract_features_matches_vector,
        test_to_grayscale_shape_and_range,
        test_real_vs_synthetic_edge_case,
        test_patch_pooling_beats_global_pooling_on_realistic_splice,
    ]
    print(f"Running {len(tests)} SRM filter bank checks...\n")
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print(f"\nAll {len(tests)} checks passed.")


if __name__ == "__main__":
    run_all()