"""Tests for edgar/dashboard/latex_cache.py.

Covers the on-demand caching behaviour and the bulk pre-render path that runs
at end-of-run for the final population. The actual LLM call is patched out so
no API key is required.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from edgar.dashboard.latex_cache import (
    _llm_from_task_spec,
    get_or_generate_latex,
    prerender_latex_for_run,
    read_cached_latex,
)


def _write_task_spec(run_dir: Path, model_llm) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_spec.yaml").write_text(
        f"llms:\n  model_llm: {json.dumps(model_llm)}\n"
    )


def test_llm_from_task_spec_handles_list(tmp_path: Path):
    """`model_llm` may be a list (cycled per generation). We need a string."""
    _write_task_spec(tmp_path, ["claude-haiku-4-5", "claude-sonnet-4-6"])
    assert _llm_from_task_spec(tmp_path / "task_spec.yaml") == "claude-haiku-4-5"


def test_llm_from_task_spec_handles_string(tmp_path: Path):
    _write_task_spec(tmp_path, "claude-haiku-4-5")
    assert _llm_from_task_spec(tmp_path / "task_spec.yaml") == "claude-haiku-4-5"


def test_llm_from_task_spec_missing(tmp_path: Path):
    assert _llm_from_task_spec(tmp_path / "nonexistent.yaml") is None


def test_llm_from_task_spec_prefers_jax_translator(tmp_path: Path):
    """jax_model_translator_llm is the primary source for LaTeX (upstream's
    choice — usually the cheaper model); it wins over model_llm."""
    (tmp_path / "task_spec.yaml").write_text(
        "llms:\n"
        f"  model_llm: {json.dumps('claude-sonnet-4-6')}\n"
        f"  jax_model_translator_llm: {json.dumps('claude-haiku-4-5')}\n"
    )
    assert _llm_from_task_spec(tmp_path / "task_spec.yaml") == "claude-haiku-4-5"


def test_llm_from_task_spec_jax_translator_list(tmp_path: Path):
    """jax_model_translator_llm may be a list (cycled per generation); the config
    schema allows list[ValidLLMs]. We need a single string for the LaTeX call."""
    (tmp_path / "task_spec.yaml").write_text(
        f"llms:\n  jax_model_translator_llm: {json.dumps(['claude-haiku-4-5', 'x'])}\n"
    )
    assert _llm_from_task_spec(tmp_path / "task_spec.yaml") == "claude-haiku-4-5"


def test_prerender_skips_already_cached(tmp_path: Path):
    """Programs with an existing cache file should be skipped, not re-rendered."""
    _write_task_spec(tmp_path, "claude-haiku-4-5")
    cache_dir = tmp_path / "latex_cache"
    cache_dir.mkdir()
    (cache_dir / "0.json").write_text(json.dumps({"latex": "$$x$$", "name": "P0"}))

    with patch("edgar.llm.llm_calling.call_llm") as mock_call:
        result = asyncio.run(
            prerender_latex_for_run(
                tmp_path,
                [(0, "P0", "def model(): pass")],
                concurrency=2,
            )
        )

    assert result["n_already_cached"] == 1
    assert result["n_generated"] == 0
    assert result["n_failed"] == 0
    mock_call.assert_not_called()


def test_prerender_renders_missing_programs(tmp_path: Path):
    """Programs without a cache file get rendered + cached."""
    _write_task_spec(tmp_path, "claude-haiku-4-5")

    async def _fake_call_llm(prompt, llm_model, output_type, temperature, retry_config):
        return f"$$y_{{{prompt.count('model')}}} = x$$"

    with patch(
        "edgar.llm.llm_calling.call_llm", side_effect=_fake_call_llm
    ) as mock_call:
        result = asyncio.run(
            prerender_latex_for_run(
                tmp_path,
                [
                    (0, "P0", "def model(): pass"),
                    (1, "P1", "def model(): pass"),
                ],
                concurrency=2,
            )
        )

    assert result["n_generated"] == 2
    assert result["n_failed"] == 0
    assert mock_call.call_count == 2
    assert read_cached_latex(tmp_path, 0) is not None
    assert read_cached_latex(tmp_path, 1) is not None


def test_prerender_skips_empty_model_code(tmp_path: Path):
    """Programs with empty model source are silently skipped."""
    _write_task_spec(tmp_path, "claude-haiku-4-5")
    with patch("edgar.llm.llm_calling.call_llm") as mock_call:
        result = asyncio.run(
            prerender_latex_for_run(
                tmp_path,
                [(0, "P0", "")],
                concurrency=2,
            )
        )
    assert result["n_skipped"] == 1
    assert result["n_generated"] == 0
    mock_call.assert_not_called()


def test_prerender_records_individual_failures(tmp_path: Path):
    """One program failing should not abort the whole bulk render."""
    _write_task_spec(tmp_path, "claude-haiku-4-5")

    async def _flaky_call_llm(prompt, **kw):
        if "FAIL_ME" in prompt:
            raise RuntimeError("simulated rate-limit")
        return "$$x$$"

    with patch("edgar.llm.llm_calling.call_llm", side_effect=_flaky_call_llm):
        result = asyncio.run(
            prerender_latex_for_run(
                tmp_path,
                [
                    (0, "P0", "def model(): pass"),
                    (1, "P1", "FAIL_ME"),
                    (2, "P2", "def model(): pass"),
                ],
                concurrency=2,
            )
        )

    assert result["n_generated"] == 2
    assert result["n_failed"] == 1
    assert read_cached_latex(tmp_path, 0) is not None
    assert read_cached_latex(tmp_path, 1) is None
    assert read_cached_latex(tmp_path, 2) is not None


def test_prerender_skips_when_no_model_llm(tmp_path: Path):
    """Without a model_llm in task_spec.yaml, the bulk render exits cleanly."""
    (tmp_path / "task_spec.yaml").write_text("io: {}\n")
    with patch("edgar.llm.llm_calling.call_llm") as mock_call:
        result = asyncio.run(
            prerender_latex_for_run(
                tmp_path,
                [(0, "P0", "def model(): pass")],
                concurrency=2,
            )
        )
    assert result["n_failed"] == 1
    assert result["n_generated"] == 0
    mock_call.assert_not_called()


def test_get_or_generate_latex_round_trip(tmp_path: Path):
    """On-demand path: first call hits LLM + writes cache; second returns cached."""
    _write_task_spec(tmp_path, "claude-haiku-4-5")
    detail = {"name": "P0", "code": {"model": "def model(): pass"}}

    async def _fake_call_llm(prompt, **kw):
        return "$$z = 1$$"

    with patch(
        "edgar.llm.llm_calling.call_llm", side_effect=_fake_call_llm
    ) as mock_call:
        first = asyncio.run(get_or_generate_latex(tmp_path, 0, detail))
        assert first["latex"] == "$$z = 1$$"
        assert first["cached"] is False
        assert mock_call.call_count == 1

        second = asyncio.run(get_or_generate_latex(tmp_path, 0, detail))
        assert second["latex"] == "$$z = 1$$"
        assert second["cached"] is True
        assert mock_call.call_count == 1  # not re-called

        forced = asyncio.run(get_or_generate_latex(tmp_path, 0, detail, force=True))
        assert forced["cached"] is False
        assert mock_call.call_count == 2


def test_get_or_generate_latex_no_model_code_raises(tmp_path: Path):
    _write_task_spec(tmp_path, "claude-haiku-4-5")
    detail = {"name": "P0", "code": {"model": ""}}
    with pytest.raises(RuntimeError, match="no model source"):
        asyncio.run(get_or_generate_latex(tmp_path, 0, detail))
