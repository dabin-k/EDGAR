# CLAUDE.md — Working conventions for this repo

How Virginia and Claude collaborate on EDGAR. Auto-loaded as context every session.
A new agent should read this + the latest `journal/YYYY-MM-DD.md` and be caught up.

---

## Where things live

| Concern | Location |
|---|---|
| Architectural map of the codebase | `overview.md` |
| What we worked on today / next | `journal/YYYY-MM-DD.md` (one file per day) |
| Working conventions (this file) | `CLAUDE.md` |
| Stable repo docs | `README.md` |
| Local-only data | `data/` (gitignored except `_generate_dummy.py`) |
| API keys | `.env` (gitignored) |
| Run outputs | `program_databases/` (gitignored) |

---

## Environment

- **Python env:** `edgar` (miniforge, Python 3.13). Activate with `conda activate edgar`.
- **Direct path** (if conda activate isn't reliable in a Claude tool call): `/opt/homebrew/Caskroom/miniforge/base/envs/edgar/bin/python`.
- **Install the repo as a package** (one-time, after activating the env):
  ```bash
  pip install -e .
  ```
  This makes `import edgar` work from any cwd / IDE cell. `pyproject.toml` at the repo root drives the install; deps are sourced dynamically from `requirements.txt` so that file stays the source of truth.
- **Dependencies:** `requirements.txt`. Notable pins/decisions:
  - `jax[cuda12]` is Linux-only; macOS gets plain `jax`. Conditional via `sys_platform` markers.
  - `pydantic-ai>=1.0` (code uses `pydantic_ai.capabilities`).
  - `anthropic` and `google-genai` are unpinned — they're transitive via pydantic-ai and old pins block resolution.
- **Package was renamed from `src` to `edgar`** (2026-05-24). Any older chat / journal references to `from src.X import Y` or `python -m src.cli` are stale; use `from edgar.X import Y` and `python -m edgar.cli` instead.

---

## How to run

- No console-script `edgar` binary is wired yet (the CLI dispatch function is `run_cli`, not `main`, so `[project.scripts]` is intentionally not set). Use:
  ```bash
  python -m edgar.cli run projects/<task>/config.yaml
  ```
- Override config from the CLI: `--evolution.n_generations=1 --io.data_path=/path.npy`.
- For a fast smoke run: `--evolution.n_generations=1 --evolution.n_islands=2 --evolution.batch_size=2 --evolution.topology="[1,0]"`.

---

## API keys

Code reads `GOOGLE_API_KEY` (not `GEMINI_API_KEY` as the README claims) for Gemini.
Once Anthropic support lands, it'll also read `ANTHROPIC_API_KEY`. Put both in `.env`:

```
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
```

**Free-tier Gemini quota is 5 req/min on `gemini-2.5-flash`** — small smoke tests exhaust it instantly. Add billing or switch model for real runs.

---

## Git workflow

- Working branch: `vmsr` (forked from `reilly_image_model_debug`).
- Main upstream is `main`; the active development branch is `reilly_image_model_debug`.
- Don't push to anything other than `vmsr` without confirming first.
- Commits: small, focused, descriptive. Use HEREDOC for multi-line commit messages.
- Don't commit `.env`, `data/*.npy`, or `program_databases/` (all gitignored).

---

## How we work

### Cadence
- Each day starts with creating `journal/YYYY-MM-DD.md` with the aims list.
- After each substantive task, update the journal's "Done" section with what changed and why.
- End-of-day: write the "Next" section so the next session (or agent) knows where to pick up.

### Before non-trivial changes
- Propose the approach in 2-3 sentences and confirm before implementing. Especially true for:
  - Anything that changes the public API of a module
  - Anything that touches `requirements.txt`
  - Anything that adds files outside the project being worked on
- For lookups / file edits / small fixes: just do it.

### Verify before claiming done
- If a code change is supposed to make something work, run it.
- If a feature isn't testable in this environment (UI, GPU, real data), say so explicitly rather than claiming success.
- Don't mark a TaskCreate task `completed` if any step failed.

### Don't
- Add CLI flags, error handling, or abstractions "for the future." Add them when needed.
- Add comments that re-state what the code does. Only comment the *why* when non-obvious.
- Bypass the user — no `--no-verify`, `--force`, or amending pushed commits without asking.
- Re-read a file Claude just edited; the harness tracks state.

### Communication
- Short responses by default. Match the question's depth.
- When proposing options, list the recommended one first with `(Recommended)`.
- Surface risks before acting on them (e.g. "this will rotate the API key", "this overwrites local changes").

---

## Conventions for this codebase

- **Project structure**: every project lives under `projects/<task>/` with the layout described in `overview.md`. Match this when adding new ones.
- **Data contract**: `load_data()` returns `(X_discover, X_validate, X_eval)` of JAX dicts; per-sample shape `(n_trials,)`. Don't deviate without updating `overview.md`.
- **Scoring is sandboxed**: every program is run in a fresh subprocess with a wall-clock timeout. Don't try to inline scoring "for speed" — LLM-generated code can hang, OOM, segfault.
- **LLM calls use `asyncio.gather(return_exceptions=True)`**: failed calls leave `None` code on the program and get filtered downstream. Don't add fail-fast behavior without thinking through how partial failures should propagate.

---

## When the session ends

1. All tasks in the in-session TaskList are either `completed` or noted in the journal under "Open / next".
2. Journal's "Next" section lists concrete handoff items (file paths, function names, exact commands).
3. If we made commits, note the SHA range in the journal.


---

## Debugging discipline

When something fails and the cause isn't immediately obvious, follow this discipline. The goal is to find the true cause, not to defend the first plausible one.

### Form hypotheses as a set, not a singleton
On the first failure, write down 2–4 candidate explanations, not one. Explicitly note which is most likely AND what the leading alternatives are. A single hypothesis is an anchor; a set keeps you honest. State your confidence in each as a rough probability.

### Every diagnostic step must be able to DISCONFIRM, not just confirm
Before running a command to investigate, ask: "What outcome would prove my leading hypothesis WRONG?" Prefer tests that can falsify over tests that merely re-observe the symptom. If a command can only ever confirm what I already believe, it's low value. The single most useful experiment is usually the one that distinguishes between two competing hypotheses in one shot — design that experiment explicitly and run it early.

### Treat surprising results as evidence, not noise to explain away
If a result contradicts my current theory (e.g. a plain `cp` fails when my theory says only one specific app should be blocked), that contradiction is a SIGNAL the theory is wrong. Do not invent a new sub-mechanism to rescue the hypothesis. Inventing a plausible-sounding rule to patch a failing theory ("X is OS-protected so only the originating app can touch it") is confabulation — flag it to myself as such and downgrade the hypothesis instead.

### Know which mechanisms are real before invoking them
Before asserting that some system behaves a certain way ("this attribute gates file access by originating process"), check: do I actually know this is how it works, or am I pattern-matching to something that sounds right? If I'm not sure, say so explicitly and verify (docs, web search, or a direct test) rather than building a chain of reasoning on an assumed mechanism. State the assumption out loud so it's auditable.

### Re-read my own evidence before concluding
Before committing to an explanation, scan back over everything observed this session. Often the disconfirming fact is already on the screen (e.g. I correctly identified the sandbox earlier, then ignored it). My own earlier output is data; don't let a later narrative overwrite an earlier correct observation.

### Separate "what changed" from "what I started doing differently"
If access/behavior seems to "suddenly" break mid-session, distinguish between (a) the environment actually changing and (b) me starting to do a different kind of operation (e.g. moving from session-created files to pre-existing ones). The second is far more common and points to a structural cause, not a transient one.

### Escape the loop after 2 failed fixes
If two attempted fixes for the same hypothesis both fail, STOP iterating on that hypothesis. Explicitly reconsider the premise. State: "My working theory is X; two fixes have failed; here are the alternative theories I deprioritized and the one cheap test that would distinguish them." Then run that test. Do not propose a third variation of the same fix.

### Prefer the test that resolves the question over the fix that assumes the answer
When blocked, the instinct is to keep trying fixes. Resist it. One well-chosen diagnostic that tells you WHICH world you're in is worth more than three speculative fixes. Cheap, decisive, falsifying — in that order.

### Don't outsource the diagnosis to the user prematurely
Before asking the user to run privileged commands or change system settings, exhaust the tests I can run myself, especially falsifying ones. If I must ask, ask for the output of a test that distinguishes hypotheses, not for a fix that presumes one.