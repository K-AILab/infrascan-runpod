"""Tests for the validation tiers."""
from __future__ import annotations

import pytest

from app import validation


def test_tier1_extension():
    assert validation.tier1_check("foo.insv", 50 * 1024 * 1024, "insta360") is None
    assert "Expected" in validation.tier1_check("foo.txt", 50 * 1024 * 1024, "insta360")


def test_tier1_size_min():
    err = validation.tier1_check("foo.mp4", 1 * 1024 * 1024, "video")
    assert err and "too small" in err


def test_tier1_size_max():
    err = validation.tier1_check("foo.mp4", 60 * 1024 * 1024 * 1024, "video")
    assert err and "too large" in err


def test_tier1_unknown_type():
    err = validation.tier1_check("foo.insv", 1024 * 1024 * 1024, "wat")
    assert err and "Unknown capture type" in err


def test_tier1_tar_gz_accepted_for_frames():
    assert validation.tier1_check(
        "bundle.tar.gz", 200 * 1024 * 1024, "frames"
    ) is None


def test_failure_hint_table_is_keyed_by_stage_name():
    # Sanity: keys are the canonical stage names the pipeline runner uses.
    for k in ("00b_gen_da3", "01_propose", "02_embed", "03_backproject", "04_index"):
        assert k in validation.FAILURE_HINTS
