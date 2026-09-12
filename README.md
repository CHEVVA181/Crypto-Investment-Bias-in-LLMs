# Crypto Investment Bias in LLMs

Master's project, Department of Informatics, University of Zurich. Replicates and
extends the concentration methodology from Zhi et al. (2025), applied to crypto.

## What this is

I asked four models — GPT-5.5, Claude Haiku 4.5, Gemini 3.6 Flash and Grok 4.6 — a big
batch of CHF-denominated prompts about which crypto tokens to buy and which exchanges to
use. This repo is the code that turns those raw responses into concentration stats,
the figures used in the write-up, and a check for whether a model favors products tied
to its own corporate parent. There's also a test suite covering the parser and the Gini
implementation, since a lot rides on both being right.

## Layout

```
src/analy_db.py               main pipeline: parse -> normalise -> concentration -> figures
src/affiliation_analysis.py   provider-affiliation / conflict-of-interest check
tests/test_analy_db.py        51 checks over parsing, units, Gini and pipeline output
```

The response databases and the generated output aren't meant to sit loose in the repo
root — see [Data](#data).

## Install

```bash
pip install -r requirements.txt
```

Needs Python 3.11+. `pygini` is only there to sanity-check the hand-written Gini
formula against a known-good implementation; the pipeline runs fine without it and just
skips that one audit.

## Running the pipeline

```bash
export CRYPTO_BIAS_DB=/path/to/responses.db      # default: <repo>/data/responses.db
python src/analy_db.py
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `CRYPTO_BIAS_DB` | SQLite file with the `responses` table | `<repo>/data/responses.db` |
| `CRYPTO_BIAS_EXTRA_DBS` | Top-up databases, `os.pathsep`-separated. `off` / `none` disables auto-discovery | auto: `responses-missing*.db` next to the main DB |
| `CRYPTO_BIAS_OUT` | Output root | `<db folder>/crypto_bias_output` |

A top-up database is just for backfilling prompts a model missed on the first
collection run. Rows are matched on `(scenario, condition, model, prompt index)` rather
than the raw id, so re-running an already-present prompt gets dropped instead of
duplicated. `data_sources.csv` in the output keeps a per-model row count of every file
that contributed, in case you need to check where a number came from.

The pipeline writes one folder per scenario (`tokens/`, `exchanges/`), each with about
20 CSVs and 7 figures, plus a `cross_scenario_gini.csv` and a `README.txt` at the output
root listing everything it just wrote.

### Provider-affiliation check

This runs on the pipeline's output, not on the raw database:

```bash
python src/affiliation_analysis.py crypto_bias_output crypto_bias_output/affiliation
```

It checks whether a model over-recommends assets or exchanges tied to its own corporate
parent (or a controlling principal of that parent). Every affiliation used here is a
publicly disclosed ownership or partnership fact that predates the collection window —
nothing inferred after the fact from the responses themselves. Significance is a
10,000-draw bootstrap with a fixed seed, so it's reproducible if you rerun it.

## Tests

```bash
python tests/test_analy_db.py
```

Builds a small synthetic SQLite fixture, runs the whole pipeline over it, and checks
what comes out. Exits 0 if all 51 checks pass, 1 otherwise. Roughly five areas:

1. **Fixture run** — pipeline imports and completes end to end for both scenarios, core CSVs get written.
2. **Parsing** — number formats it should accept (`1'500`, `1 500`, `1.000,50`, `1,000.50`), and just as important, what it should refuse: `0.05 BTC`, `3 years`, `1/3 of portfolio`, a bare digit sitting in prose, `40 60` (a space only groups thousands in threes — gluing that into `4060` would be inventing a number).
3. **Units** — `40%` means 40% of the money to invest, never 40 CHF flat.
4. **Gini** — the closed-form properties, agreement with `pygini`, and the own-vs-union support behaviour.
5. **Output** — what the pipeline actually does with the fixture: refusals produce no product, `Bitcoin` and `BTC` collapse into one row, a stated range becomes its midpoint, a zero allocation counts as a rejection, and every response's shares sum to 1.

Test artefacts land in `tests/test_output/` and are gitignored, so they won't clutter a
diff.

## Method notes

A few judgment calls that are easy to get wrong, so they're written down here instead of
just living in a comment somewhere:

**Gini support.** GI follows Zhi et al. (2025) eq. 1, computed over the union support —
every product any model named in that scenario, zero-filled for models that didn't name
it. That keeps `n` and the product set identical across models, so the numbers are
actually comparable. Scoring a model only against its own list (the paper's default)
flatters a narrow model into looking artificially equal to the others; those values are
still kept around as `GI_*_own`. Swap the headline metric with `GINI_SUPPORT` at the top
of `analy_db.py`.

**Amounts.** A percentage is a share of the money to invest. Given a stated budget,
`40%` of `10'000 CHF` is `4'000 CHF`. Without a stated budget the split is still exact,
just not sized in CHF — no value gets invented (`unit_basis='proportional'`). A bare
number is read as either percent or francs depending on the rest of the list. If an
answer mixes both without stating a budget, the pot gets inferred from the franc
amounts (`unit_basis='implied'`).

**Currency.** Prompts are in CHF, but models sometimes answer in USD or EUR anyway. The
currency actually written is recorded per row rather than silently relabelled to CHF,
and no FX conversion is applied — a constant factor would cancel out of the shares
anyway. Budget utilisation just doesn't get computed for those rows.

**A list isn't necessarily an allocation.** If three products each get the full stated
budget, that reads as "put your money into any of these," not "invest three times the
budget." Those responses get flagged `flat_full_budget` and scored on rank order instead
(`FLAT_BUDGET_AS_ALTERNATIVES`).

**The refusal confound.** The balanced panel controls for how many prompts each model
was *asked*, not how many it actually *answered*. A model that refuses more ends up
naming fewer distinct products, and a shorter list over the same union support scores as
more concentrated almost by construction — so refusal rate and measured GI are tangled
together. `gini_paired_usable.csv` re-scores every model only on the prompts every model
answered, and reports whether the ranking still holds up.

## Data

The response databases live in `data/`: `responses.db` is the main collection run, and
`data/responses-missing_grok.db` is a top-up that backfills token-scenario prompts Grok
missed the first time around. `analy_db.py` merges them automatically — see
[Running the pipeline](#running-the-pipeline). The generated `crypto_bias_output/` tree
isn't tracked.

Every model got asked the same 719 prompts per scenario:

| Model | Tokens prompts | Exchanges prompts |
| --- | ---: | ---: |
| GPT-5.5 | 719 | 719 |
| Claude Haiku 4.5 | 719 | 719 |
| Gemini 3.6 Flash | 719 | 719 |
| Grok 4.6 | 719 | 719 |

Grok's token count only hits 719 once `responses-missing_grok.db` gets merged in (515
from the main run, plus a deduplicated top-up). Every other cell is already 719 in
`responses.db` on its own.

### How the prompts are built

Each prompt states a budget (CHF), a risk tolerance, an investment term and a market
environment, drawn from these values:

| Attribute | Values | Count |
| --- | --- | ---: |
| budget | 100; 1,000; 10,000; 20,000; 30,000; 40,000; 50,000; 100,000 CHF | 8 |
| risk | risk-averse, risk-neutral, risk-seeking | 3 |
| term | less than one year, one to three years, three to ten years | 3 |
| environment | crisis, recession, recovery, expansion | 4 |

No prompt varies all four attributes independently at once. Instead, each one sweeps
every value of some subset of the four attributes while holding the rest fixed at a
baseline, and the database's `variables` column records which subset that was. Add up
all 15 non-empty subsets of the four attributes and you get the full 719-prompt panel —
same breakdown for `tokens` and `exchanges`:

| Attributes varied | Values combined | Prompts |
| --- | --- | ---: |
| budget | 8 | 8 |
| risk | 3 | 3 |
| term | 3 | 3 |
| environment | 4 | 4 |
| budget × risk | 8 × 3 | 24 |
| budget × term | 8 × 3 | 24 |
| budget × environment | 8 × 4 | 32 |
| risk × term | 3 × 3 | 9 |
| risk × environment | 3 × 4 | 12 |
| term × environment | 3 × 4 | 12 |
| budget × risk × term | 8 × 3 × 3 | 72 |
| budget × risk × environment | 8 × 3 × 4 | 96 |
| budget × term × environment | 8 × 3 × 4 | 96 |
| risk × term × environment | 3 × 3 × 4 | 36 |
| budget × risk × term × environment | 8 × 3 × 3 × 4 | 288 |
| **Total** | | **719** |

Expected schema:

```sql
CREATE TABLE responses (
    id TEXT PRIMARY KEY, scenario TEXT, variables TEXT, model TEXT NOT NULL,
    model_version TEXT NOT NULL, response_timestamp TEXT NOT NULL,
    prompt TEXT NOT NULL, response TEXT NOT NULL)
```

## Reference

Zhi et al. (2025) — the concentration methodology this project replicates and extends.
