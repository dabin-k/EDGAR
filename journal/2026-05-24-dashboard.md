# 2026-05-24 — Dashboard

Built a unified live + inspect dashboard from first principles, replacing the
three standalone HTML reports under the old `edgar/monitoring/`.

## Stack
- **Backend:** FastAPI on `127.0.0.1:8765`. Pure file-reader; no shared-memory
  IPC with the runner. Pydantic-free DTOs (plain dicts) for low ceremony.
- **Frontend:** single static `index.html` + `app.js` + `styles.css` under
  `edgar/dashboard/static/`. Alpine.js for reactivity, Plotly for charts,
  KaTeX for math, highlight.js for code, Tailwind via Play CDN — no build step.
- **Live updates:** 2.5s polling of `/api/runs/{id}/state`. Cadence matched to
  generation length (~30s+), so WebSockets / SSE would be over-engineering.

## Runner change

`edgar/run.py` now writes per-generation snapshots of `population.jsonl` +
`island_census.jsonl`, and emits `status.json` with state transitions:

```
starting -> running (per generation) -> complete | failed
```

All writes are atomic (`.tmp.<pid>.<tid>` + `os.replace`). The dashboard
treats `status.state == 'running' && now - updated_at > 60s` as `'failed'`
(`is_stale=True`) so SIGKILL'd runs surface clearly rather than appearing
to hang.

`edgar/monitoring/` was deleted; `tutorials/inspect_outputs.py` cell 12 now
demonstrates `edgar.dashboard` programmatically instead.

## CLI

```
python -m edgar.cli dashboard                       # picker over program_databases/
python -m edgar.cli dashboard <run_dir>             # opens that run directly (Inspect)
python -m edgar.cli dashboard --port 8765 --no-open
```

## API

- `GET /api/runs` — list summary cards
- `GET /api/runs/{id}/summary` — task name, n_islands, n_gens, LLMs, the prompt text
- `GET /api/runs/{id}/state` — live state: islands × programs × best/elapsed/eta + log tail
- `GET /api/runs/{id}/programs` — ranked program list
- `GET /api/runs/{id}/programs/{idx}` — detail: code, params, lineage
- `POST /api/runs/{id}/programs/{idx}/latex` — LLM-derived LaTeX, cached under `<run_dir>/latex_cache/`
- `GET /api/runs/{id}/image/gen_{g}/island_{i}/batch_{b}` — model-fit images

## Testing

`tests/dashboard/` has 12 tests covering:
- Endpoint shape against the existing run at `program_databases/05-24/17-17-45/`
- Legacy-run tolerance (no `status.json` ⇒ implicit `complete`)
- Atomic-write invariant: 4 readers vs 2 writers for 1.5s, zero `JSONDecodeError`
- LaTeX cache: first call hits a (monkey-patched) LLM and writes cache; second
  reads from cache; `force=true` re-derives
- Stale-run detection: synthetic `state='running'` with old `updated_at`
  surfaces as `failed` + `is_stale=True`
- Per-generation persistence via `edgar test-fake`: `status.json` transitions
  `starting → running → complete`, `population.jsonl` grows monotonically
- Failure path via synthetic exception in `score()`: `status.json` ends at
  `failed` with the exception name in `.error`

Manual smoke tests against the existing 17-17-45 run and a fresh `edgar test`
run both pass. 5 Hz HTTP polling of `/api/state` against a live run for 30s
produced 272 successful responses, 0 errors. Real `claude-haiku-4-5` LaTeX
generation took 3.7s on the rank-1 winner; cache hit returns in 3 ms.

## Files added / changed

- `edgar/dashboard/{__init__.py,data.py,server.py,latex_cache.py}`
- `edgar/dashboard/static/{index.html,app.js,styles.css}`
- `edgar/io/status.py` (new: atomic file helpers + status.json schema)
- `edgar/run.py` (per-gen persist + status transitions)
- `edgar/cli.py` (`dashboard` subcommand)
- `edgar/evolution/population.py` (Population.save uses atomic_write_text)
- `edgar/evolution/island.py` (save_island_census uses atomic_write_text)
- `requirements.txt` (`fastapi>=0.110`, `uvicorn[standard]>=0.27`)
- `pyproject.toml` (package-data for `edgar.dashboard/static/`)
- `tests/dashboard/{__init__.py,test_api.py,test_runner_persist.py}`
- `tutorials/inspect_outputs.py` (cell 12 rewritten)
- `tutorials/walkthrough_orientation_tuning.py` (drop `write_family_tree`)
- `tutorials/how_to_run.md` (per-gen persistence note)
- `overview.md` (output dir layout + dashboard command)

Deleted: `edgar/monitoring/` entirely.

## Open / next

- The dashboard polls every 2.5s; we could turn that down to 1s safely.
- LaTeX cache is per-program JSON files; for tens of thousands of programs
  this would benefit from a single SQLite file. Not urgent.
- The seed phase shows `current_gen=-1` in the live view, which the UI
  currently renders as "Generation 0 / N" via `(-1 + 1)`. Fine but worth a
  pass: maybe show "seeding" explicitly.
- Two simultaneous live runs work via the run picker. No special handling
  beyond that.
