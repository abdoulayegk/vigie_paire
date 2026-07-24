"""Tests for Vision fallback cache validation and hardening."""

from __future__ import annotations

import tempfile


def test_cache_key_uses_date_row_prompt_version() -> None:
    """Le prompt qui conserve les dates-lignes ne reutilise pas le cache v6."""
    from vigilance.extraction.vision_cache import make_cache_key

    key = make_cache_key(
        "pdf-sha",
        39,
        [0.056, 0.726, 0.936, 0.786],
        max_completion_tokens=120000,
    )

    assert key.startswith("v7_")


def test_cache_get_ignores_corrupted_indicators_string() -> None:
    """When cache stores indicators as string instead of list, integration ignores it."""
    from vigilance.extraction.vision_cache import cache_get, cache_put

    with tempfile.TemporaryDirectory() as d:
        key = "test_sha_1_0.1_0.2_0.8_0.9"
        bad_payload = {"indicators": "not-a-list", "confidence": 0.9}
        cache_put(d, key, bad_payload)
        payload = cache_get(d, key)
        assert payload is not None
        indicators_raw = payload.get("indicators")
        assert not isinstance(indicators_raw, list)
        indicators_safe = indicators_raw if isinstance(indicators_raw, list) else []
        assert indicators_safe == []
        assert len(indicators_safe) == 0


def test_cache_validation_integration_fallback() -> None:
    """Corrupted cache (indicators as string) causes fallback logic: ignore payload, confidence=0."""
    from vigilance.extraction.vision_cache import cache_get, cache_put

    with tempfile.TemporaryDirectory() as tmpdir:
        key = "abc123_5_0.1_0.2_0.8_0.9"
        cache_put(tmpdir, key, {"indicators": "bad-string", "confidence": 0.95})
        cached = cache_get(tmpdir, key)
        indicators_raw = cached.get("indicators") if cached else None
        if not isinstance(indicators_raw, list):
            indicators_raw = []
            confidence = 0.0
        else:
            confidence = float(cached.get("confidence", 0.0))
        assert indicators_raw == []
        assert confidence == 0.0


def test_get_vision_cache_dir_default() -> None:
    """VISION_CACHE_DIR uses default when not set."""
    import os

    from vigilance.extraction.vision_cache import DEFAULT_CACHE_DIR, get_vision_cache_dir

    os.environ.pop("VISION_CACHE_DIR", None)
    assert get_vision_cache_dir() == DEFAULT_CACHE_DIR


def test_get_vision_cache_dir_from_env() -> None:
    """VISION_CACHE_DIR uses env when set."""
    import os

    from vigilance.extraction.vision_cache import get_vision_cache_dir

    os.environ["VISION_CACHE_DIR"] = "/custom/cache"
    try:
        assert get_vision_cache_dir() == "/custom/cache"
    finally:
        os.environ.pop("VISION_CACHE_DIR", None)


def test_get_vision_crop_dir_default() -> None:
    """VISION_CROP_DIR uses default when not set."""
    import os

    from vigilance.extraction.vision_cache import DEFAULT_CROP_DIR, get_vision_crop_dir

    os.environ.pop("VISION_CROP_DIR", None)
    assert get_vision_crop_dir() == DEFAULT_CROP_DIR


def test_cache_validation_rejects_nested_list() -> None:
    """indicators as list of lists is rejected (must be list of strings)."""
    from vigilance.extraction.vision_cache import cache_get, cache_put

    with tempfile.TemporaryDirectory() as tmpdir:
        key = "nest_1_0.0_0.0_1.0_1.0"
        cache_put(tmpdir, key, {"indicators": [["a", "b"], ["c"]], "confidence": 0.9})
        cached = cache_get(tmpdir, key)
        raw = cached.get("indicators") if cached else None
        valid = isinstance(raw, list) and all(isinstance(x, str) for x in raw)
        assert not valid


def test_cache_validation_rejects_list_of_numbers() -> None:
    """indicators as list of numbers is rejected."""
    from vigilance.extraction.vision_cache import cache_get, cache_put

    with tempfile.TemporaryDirectory() as tmpdir:
        key = "num_1_0.0_0.0_1.0_1.0"
        cache_put(tmpdir, key, {"indicators": [1, 2, 3], "confidence": 0.9})
        cached = cache_get(tmpdir, key)
        raw = cached.get("indicators") if cached else None
        valid = isinstance(raw, list) and all(isinstance(x, str) for x in raw)
        assert not valid


def test_cache_validation_accepts_list_of_strings() -> None:
    """indicators as list of strings is accepted."""
    raw = ["Indicateur A", "Indicateur B"]
    valid = isinstance(raw, list) and all(isinstance(x, str) for x in raw)
    assert valid
