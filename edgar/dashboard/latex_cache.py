"""latex_cache.py — on-demand LaTeX equation rendering for evolved programs.

This module is responsible for translating the numpy `model` source code of
an evolved program into its mathematical LaTeX representation using an LLM.
The generated LaTeX is aggressively cached on disk to ensure that subsequent
requests are fast and do not incur additional LLM costs.

The process involves:
1. Checking for a cached LaTeX representation for a given program.
2. If not found or forced, extracting the program's numpy model source code.
3. Identifying the specific LLM (the `jax_model_translator_llm` from the run's
   `task_spec.yaml`) that was configured for JAX model translation.
4. Constructing a prompt (`_LATEX_PROMPT`) to instruct the LLM to output
   only the LaTeX equations.
5. Calling the LLM via `llm_calling.call_llm` with a low temperature for
   deterministic output and retries for robustness.
6. Atomically writing the LLM's LaTeX response to a JSON file
   (`run_dir/latex_cache/{idx}.json`) for persistent caching.

This mechanism ensures that the dashboard can display mathematical equations
for evolved models efficiently without re-invoking the LLM unless necessary.

Reused with light edits from the pattern in tutorials/inspect_outputs.py.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable

import yaml

from ..io.config import RetryConfig
from ..io.status import atomic_write_text


CACHE_DIRNAME = "latex_cache"


def _cache_path(run_dir: Path, idx: int) -> Path:
    """Constructs the file path for a cached LaTeX equation.

    Args:
        run_dir: The root directory of the EDGAR run.
        idx: The global index of the program.

    Returns:
        The full path to the cached LaTeX JSON file for the given program.
    """
    return Path(run_dir) / CACHE_DIRNAME / f"{idx}.json"


def read_cached_latex(run_dir: Path, idx: int) -> dict | None:
    """Reads a cached LaTeX equation from disk.

    Checks if a cached LaTeX file exists for a given program index within
    the specified run directory and attempts to load its content.

    Args:
        run_dir: The root directory of the EDGAR run.
        idx: The global index of the program.

    Returns:
        A dictionary containing the cached LaTeX data if found and successfully
        parsed, otherwise `None`.
    """
    path = _cache_path(run_dir, idx)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is corrupted or not valid JSON, treat as if not cached.
        return None


async def get_or_generate_latex(
    run_dir: Path,
    idx: int,
    program_detail: dict,
    force: bool = False,
) -> dict:
    """Returns cached LaTeX if present, otherwise generates and caches it using an LLM.

    This function first attempts to read the LaTeX representation of a program's
    model from a local cache. If no cached version is found, or if `force` is
    set to `True`, it invokes an LLM to generate the LaTeX from the program's
    numpy model source code. The generated LaTeX is then stored in the cache
    for future requests.

    Args:
        run_dir: The root directory of the EDGAR run.
        idx: The global index of the program for which to get or generate LaTeX.
        program_detail: A dictionary containing the program's details,
            specifically its model source code under `code.model` and its name.
        force: If `True`, forces the regeneration of LaTeX via the LLM,
            bypassing the cache. Defaults to `False`.

    Returns:
        A dictionary containing the LaTeX string and associated metadata,
        including an `idx`, `name`, the `llm` model used, the `latex` string
        itself, `generated_at` timestamp, and a `cached` boolean flag indicating
        if the result was from cache (`True`) or newly generated (`False`).

    Raises:
        RuntimeError: If the program has no model source code, if no
            `jax_model_translator_llm` is found in the `task_spec.yaml`,
            if the LLM call fails (e.g., missing API key, quota issues,
            network errors), or if the LLM returns an empty response.
    """
    run_dir = Path(run_dir)
    if not force:
        cached = read_cached_latex(run_dir, idx)
        if cached is not None:
            return {**cached, "cached": True}

    model_code = (program_detail.get("code") or {}).get("model") or ""
    if not model_code:
        raise RuntimeError(f"program {idx} has no model source")

    name = program_detail.get("name") or f"P{idx}"

    spec_path = run_dir / "task_spec.yaml"
    llm_model = _llm_from_task_spec(spec_path)
    if not llm_model:
        raise RuntimeError(
            "no model_llm found in task_spec.yaml; can't pick an LLM for LaTeX generation"
        )

    prompt = _LATEX_PROMPT.format(name=name, code=model_code)

    try:
        from ..llm.llm_calling import call_llm
    except ModuleNotFoundError as e:
        import sys

        raise RuntimeError(
            f"LLM dependencies are missing in {sys.executable!r} "
            f"(failed to import {e.name!r}). This is likely due to running the "
            "dashboard from the wrong environment. Activate the 'edgar' conda env, "
            "`pip install -e .` from the repo root, or use the prefix `uv run` "
            "and restart the dashboard."
        ) from e
    try:
        retry_config = RetryConfig()
        latex = await call_llm(
            prompt=prompt,
            llm_model=llm_model,
            output_type=str,
            temperature=0.2,  # Low temperature for deterministic output
            retry_config=retry_config,
        )
    except Exception as e:  # noqa: BLE001 — surface everything as a clean error
        raise RuntimeError(f"LLM call failed: {type(e).__name__}: {e}") from e

    if not latex:
        raise RuntimeError("LLM returned an empty LaTeX response")

    payload = {
        "idx": idx,
        "name": name,
        "llm": llm_model,
        "latex": latex,
        "generated_at": time.time(),
    }
    cache_path = _cache_path(run_dir, idx)
    atomic_write_text(cache_path, json.dumps(payload, indent=2))
    return {**payload, "cached": False}


async def prerender_latex_for_run(
    run_dir: Path,
    programs: list[tuple[int, str, str]],
    *,
    concurrency: int = 8,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    """Bulk-fill the LaTeX cache for a list of programs.

    Skips programs that already have a cache file. Programs without model
    source code are skipped silently. Individual failures are logged but do
    not raise — the on-demand `get_or_generate_latex` path still works as a
    fallback for anything that fails here.

    Args:
        run_dir: run directory containing task_spec.yaml.
        programs: list of (idx, name, model_code) tuples. Empty model_code is
            silently skipped.
        concurrency: max simultaneous LLM calls. 8 is comfortable for
            Anthropic/Google rate limits at default run settings.
        log_fn: optional callback for progress messages (e.g. print_and_log).

    Returns:
        dict with counts: {"n_total", "n_already_cached", "n_generated",
        "n_failed", "n_skipped"}.
    """
    run_dir = Path(run_dir)

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    spec_path = run_dir / "task_spec.yaml"
    llm_model = _llm_from_task_spec(spec_path)

    to_render: list[tuple[int, str, str]] = []
    n_already_cached = 0
    n_skipped = 0
    for idx, name, model_code in programs:
        if not model_code:
            n_skipped += 1
            continue
        if read_cached_latex(run_dir, idx) is not None:
            n_already_cached += 1
            continue
        to_render.append((idx, name, model_code))

    counts = {
        "n_total": len(programs),
        "n_already_cached": n_already_cached,
        "n_generated": 0,
        "n_failed": 0,
        "n_skipped": n_skipped,
    }

    if not to_render:
        _log(
            f"[prerender_latex] nothing to do "
            f"(total={counts['n_total']} cached={n_already_cached} skipped={n_skipped})"
        )
        return counts

    if not llm_model:
        _log(
            "[prerender_latex] skipped: no model_llm in task_spec.yaml; "
            "LaTeX will be rendered on-demand from the dashboard instead."
        )
        counts["n_failed"] = len(to_render)
        return counts

    _log(
        f"[prerender_latex] rendering {len(to_render)} programs "
        f"({n_already_cached} already cached, {n_skipped} skipped) "
        f"using {llm_model}, concurrency={concurrency}..."
    )

    sem = asyncio.Semaphore(concurrency)

    async def _one(
        idx: int, name: str, model_code: str
    ) -> tuple[int, bool, str | None]:
        async with sem:
            detail = {"name": name, "code": {"model": model_code}}
            try:
                await get_or_generate_latex(run_dir, idx, detail)
                return (idx, True, None)
            except Exception as e:  # noqa: BLE001 — best-effort, never raise
                return (idx, False, f"{type(e).__name__}: {e}")

    results = await asyncio.gather(
        *[_one(idx, name, code) for idx, name, code in to_render]
    )
    for idx, ok, err in results:
        if ok:
            counts["n_generated"] += 1
        else:
            counts["n_failed"] += 1
            _log(f"[prerender_latex] program {idx} failed: {err}")

    _log(
        f"[prerender_latex] done: generated={counts['n_generated']} "
        f"failed={counts['n_failed']} cached={n_already_cached}"
    )
    return counts


def _llm_from_task_spec(spec_path: Path) -> str | None:
    """Extracts the LLM model name used for JAX model translation from the `task_spec.yaml`.

    This helper function reads the `task_spec.yaml` file from a run directory
    and retrieves the specific LLM configured for JAX model translation,
    which is then reused for LaTeX generation to maintain consistency.

    Args:
        spec_path: The file path to the `task_spec.yaml` for a given EDGAR run.

    Returns:
        The name of the LLM model (e.g., "gemini-pro") as a string, or `None`
        if the file does not exist, cannot be parsed, or the specific LLM
        configuration is not found.
    """
    if not spec_path.exists():
        return None
    try:
        with open(spec_path) as f:
            spec = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return None
    llms = spec.get("llms") or {}
    # Derive LaTeX from the jax translator LLM (upstream's choice — typically the
    # cheaper model), falling back to model_llm. Either may be a list (cycled per
    # generation); pick the first valid entry.
    model = llms.get("jax_model_translator_llm") or llms.get("model_llm")
    if isinstance(model, list):
        return model[0] if model else None
    return model


_LATEX_PROMPT = """\
You are given the numpy source of a parametric model. Output ONLY the LaTeX
equations the code implements - no prose, no explanation, no code fences.
Wrap display equations in $$...$$ so they render in Markdown. Define every
symbol you use in a brief variable-key block after the equations.

Model: {name!r}

```python
{code}
```
"""
