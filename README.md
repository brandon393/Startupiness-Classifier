# Startupiness Classifier

A two-stage pipeline at Headline VC that classifies companies into **early**, **growth**, **established**, or **non-startup**, used to support deal-flow analysis and reduce manual triage time.

## How it works

The pipeline has two stages:

1. **Cantera smart-tag queries** (Searchlight) — four deterministic queries pre-filter companies into early/growth/established/auto-exclude buckets based on hard rules (funding thresholds, round types, age, headcount). Anything that doesn't cleanly fit one of these buckets is marked "undetermined" and passed to stage two.
2. **LLM classifier** (`Startupiness_LLM_Classifier.py`) — handles everything the Cantera queries couldn't confidently resolve, using a structured prompt with explicit classification rules, tie-breaking heuristics, and a web-search fallback for ownership/subsidiary checks.

## Cantera queries

The four queries are saved as raw text (no JSON wrapper) directly in Searchlight's smart tag editor. A few syntax quirks worth knowing if you ever need to edit or recreate them:

- Round types must be written as `venture-a` / `venture-b` / `venture-c` / `venture-d` — not `series-a` or `"Series A"`, despite that being the round's display name elsewhere in Searchlight.
- Use `-` for NOT, not the word `NOT`.
- Spacing inside filters: `{type="venture-a"}` works; `{ type="venture-a" }` may not.
- `crunchbase:funding-total-usd` is the company's total funding; `crunchbase:rounds:raised-usd` is per-round.
- **Always paste raw query text directly into the smart tag editor.** Pasting JSON or text containing `\n` has corrupted saved smart tags in the past.

### Stage 1a — Early
Requires a `crunchbase:uuid`, funding under $30M, excludes venture-a/b/c/d rounds, requires a positive signal (seed/pre-seed/angel/series-unknown round, or an accelerator investor with no later round), founded within 10 years, last round within 6 years (2,190 days), excludes `.gov`/`.edu`/`.org`.

### Stage 1b — Growth
Requires a `crunchbase:uuid`, funding ceiling of $500M, matches venture-a/b/c/d rounds OR $30M–$500M total funding OR (50+ headcount AND founded under 15 years ago), last round within 5 years, excludes `.gov`/`.edu`/`.org`.

### Stage 1c — Established
IPO'd or acquired companies, OR old and unfunded, OR large and unfunded, OR funding over $500M (this last condition catches well-funded private companies like Anthropic or OpenAI that haven't IPO'd).

### Stage 1d — Auto-exclude
Terminated companies, `.gov`, `.edu`, and adult-content domains.

**The boundary logic in plain terms:** early = under $30M and pre-Series A; growth = $30M+ or Series A+, up to $500M; established = $500M+ or public/acquired.

### Known limitation: Searchlight domain misattribution

A handful of domains (`google.co.uk`, `a.co`, `manus.space`, `wf.com`, `isaac.earth`, and others) are mapped in Searchlight's underlying database to unrelated small companies rather than their actual well-known owners. No query or prompt logic can fix this — it's a source-data issue, not a classification issue. This has been flagged to the Searchlight team and is tracked as a known, accepted limitation rather than something the classifier can be expected to get right.

## LLM classifier

`Startupiness_LLM_Classifier.py` fetches companies from Searchlight via MCP (by stream token, list token, or explicit domain list) and classifies each one using an LLM, with a deterministic rules-engine fallback if the LLM call fails outright.

### Supported providers

| Provider | Flag | Default model | Notes |
|---|---|---|---|
| OpenAI | `--provider openai` (default) | `gpt-5.4-mini` | Supports web-search-grounded subsidiary checks via the Responses API |
| Gemini | `--provider gemini` | `gemini-3-flash-preview` | Requires `google-genai`; free tier is rate-limited to a small number of requests per minute and per day — see Cost & rate limits below |

Override the model for either provider with `--model`, e.g. `--provider openai --model gpt-5.4`.

### Setup

```bash
pip install requests python-dotenv openai google-genai
```

Create a `.env` file in the project root:

```
SEARCHLIGHT_TOKEN=mcp_...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

`GEMINI_API_KEY` is only required if you use `--provider gemini`. **Never commit `.env`** — it's covered by `.gitignore`, and credentials should never be hardcoded into the script itself (this has happened by accident more than once during development — always double-check before committing that `SEARCHLIGHT_TOKEN`, `OPENAI_API_KEY`, and `GEMINI_API_KEY` are reading from `os.getenv()`, not pasted in as literal strings).

`SEARCHLIGHT_TOKEN` expires periodically. If you see a `Token expired` error, generate a new one from Searchlight → Settings → MCP Server → API Tokens and update `.env`.

### Usage

```bash
python Startupiness_LLM_Classifier.py --stream <access_token>
python Startupiness_LLM_Classifier.py --list <access_token>
python Startupiness_LLM_Classifier.py --domains stripe.com notion.so linear.app
python Startupiness_LLM_Classifier.py --summary
python Startupiness_LLM_Classifier.py --stream <access_token> --provider gemini --model gemini-3-flash-preview
```

Useful flags:
- `--output <path>` — write to a specific file instead of the default `results.jsonl`
- `--clear` — delete any existing results at the output path before running (use this whenever rerunning the same domains, otherwise the script will skip them as "already classified")
- `--no-resume` — disable the default resume-by-domain behavior for a single run without deleting prior output

### Classification rules

The prompt applies rules in priority order:

1. **Parent company / subsidiary check** — divisions or subsidiaries of large public companies are classified `established`, even if they have their own domain and team.
2. **Corporate open-source check** — open-source projects maintained by a large company (e.g. `go.dev` → Google, `playwright.dev` → Microsoft, `react.dev` → Meta) are `established`, not `early`, regardless of how small the listed team looks.
3. **Tie-breaking** — when genuinely uncertain between categories, the prompt currently resolves toward `early` (see Known issues below).
4. **Stage guidance** — `early` requires pre-Series A and under $30M total funding; `growth` requires Series A+ or $30M+ total funding, regardless of headcount.

### Web search fallback

Both OpenAI and Gemini paths support a subsidiary-check fallback that re-classifies a company using live web search when the standard classification result shows specific risk signals (ownership language in the description or in Searchlight/Attio comments, a flagged acquisition, very small team with no funding and significant age, or an ambiguous services/consulting description). This roughly doubles the cost of those specific calls but meaningfully improves accuracy on ownership-related edge cases — see Known issues for a caveat on data completeness here.

## Evaluation methodology

Two independent 99-domain manually-labeled test sets are used as ground truth:

- **Luis's list** — contributed by Luis Schmitz
- **Anaphora list**

`compare_results.py` computes a deterministic 4×4 confusion matrix (early / growth / established / non-startup) and a full mismatch list directly from a manual ground-truth CSV and a classifier results file (`.jsonl` or `.csv`). It has no AI involvement — every number it produces comes straight from `pandas`-equivalent set comparisons, so results are exactly reproducible by anyone who runs it.

```bash
python compare_results.py <manual.csv> <results.jsonl> [--mismatches-out mismatches.csv]
```

### Results summary (mini vs. GPT-5.4 vs. Gemini)

| Model | Luis's list | Anaphora |
|---|---|---|
| GPT-5.4-mini | ~85–92% (varies by run) | ~88–93% (varies by run) |
| GPT-5.4 (standard) | ~81–83% | ~83–84% |
| Gemini 3 Flash Preview | qualitatively consistent with the same failure pattern (full batch run not completed — see Known issues) |

**Key finding: the smaller, cheaper `gpt-5.4-mini` model outperforms the larger `gpt-5.4` model on this task, at roughly half the per-token cost.** This was confirmed across both test lists and multiple reruns of each. The larger model does not fix the dominant error mode and introduces a secondary error mode of its own (see below).

### Known issues

**1. Tie-breaking rule over-triggers on non-investable niche tools.** The dominant, reproducible error across every model tested (mini, GPT-5.4, Gemini) is the same: small hobby tools, niche utilities, and similar non-traditional-but-technically-real products get classified as `early` when they should be `non-startup`. Confirmed failing identically across all three models on domains including `moxfield.com`, `qrfy.io`, `tradeit.gg`, `rainbet.com`, `guns.lol`, `isaac.earth`, and others. This is the highest-priority, most actionable open issue — it's a prompt-logic problem, not a model-capability problem, since three architecturally different models fail on the same domains for the same stated reasoning.

**2. GPT-5.4 (standard) shows a secondary, model-specific blind spot for non-US institutional domains** — government agencies, standards bodies, and universities (`nims.go.jp`, `open-mpi.org`, `whatwg.org`, `utsunomiya-u.ac.jp`) are occasionally misclassified as `non-startup` rather than `established`. This pattern did not appear in `gpt-5.4-mini` runs on the same domains.

**3. Run-to-run variance exists independent of the above.** Rerunning the same model against the same domain list with an identical prompt produces a different accuracy figure each time (observed spread: ~3–7 percentage points for mini, ~1 point for GPT-5.4). Roughly half of any given run's mismatches are the same domains every time (structural, see #1 and #2); the rest vary run to run and appear to be genuine LLM sampling noise rather than anything fixable via prompt changes.

**4. Gemini full-batch testing is blocked by free-tier quota.** `gemini-3-flash-preview` is limited to roughly 5 requests/minute and 20 requests/day on the free tier — confirmed via Google AI Studio's rate-limit dashboard, not just documentation. This makes a full 99+ domain comparable run impractical without enabling billing on the Gemini project. Manual spot-checks via Google AI Studio (same system prompt, same company data, temperature 0.1) confirm Gemini exhibits the same tie-breaking failure as mini and GPT-5.4 on the domains tested, but a full quantitative accuracy figure has not been collected.

### Open questions / next steps

- Revise the tie-breaking rule (issue #1 above) — likely the single highest-leverage fix available, since it's confirmed structural and reproducible across all three models tested.
- Decide whether to add a scoped `lookup_company` call for domains that already trigger the subsidiary-check path, to close the Attio-comment data gap (issue #4).
- Consider whether the GPT-5.4 institutional-domain blind spot (issue #2) warrants a more explicit rule (e.g. ".gov / .ac / standards-body domains → established") rather than relying on the model to infer it.
- If a full Gemini comparison is wanted, enabling billing on the Gemini API project would be required to get past the free-tier quota wall.

## Repository structure

```
Startupiness_LLM_Classifier.py   — main classification pipeline (OpenAI / Gemini)
compare_results.py                — deterministic confusion matrix + mismatch report
results.jsonl                     — most recent classifier output (gitignored if large)
.env                               — credentials (gitignored, never commit)
README.md                         — this file
```

## Contributors

Built by Brandon Chuang at Headline VC. Manual ground-truth labeling and ongoing review by Luis Schmitz.
