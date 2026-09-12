# Crypto Investment Bias in LLMs

Product bias in large language model investment recommendations.

Four models — **GPT-5.5**, **Claude Haiku 4.5**, **Gemini 3.6 Flash** and **Grok 4.6** —
were asked, in CHF-denominated prompts, which crypto **tokens** to invest in and which
**exchanges** to use. This repository holds the analysis code that turns those raw
responses into concentration statistics, paper-style figures and a provider-affiliation
test, plus the test suite that guards the parser and the Gini implementation.

The method replicates and extends Zhi et al. (2025). Master's project, Department of
Informatics, University of Zurich.

---

## Layout

```
src/analy_db.py               main pipeline: parse -> normalise -> concentration -> figures
src/affiliation_analysis.py   provider-affiliation / conflict-of-interest analysis
tests/test_analy_db.py        51 checks over parsing, units, Gini and pipeline output
```

The response databases and the generated outputs are not tracked here — see
[Data](#data) below.

## Install

```bash
pip install -r requirements.txt
```

Python 3.11+. `pygini` is only used to cross-check the hand-written Gini formula; the
pipeline runs without it and simply skips that audit.

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

A top-up database backfills prompts a model missed on the first collection run. Identity
is `(scenario, condition, model, prompt index)` rather than the raw row id, so re-running
an already-present prompt is dropped instead of duplicated; `data_sources.csv` records
the per-model row count of every file that contributed.

The pipeline writes one folder per scenario (`tokens/`, `exchanges/`), each with ~20 CSVs
and 7 figures, plus `cross_scenario_gini.csv` and a `README.txt` at the output root that
documents every file it just wrote.

### Provider-affiliation analysis

Runs on the pipeline's output, not on the database:

```bash
python src/affiliation_analysis.py crypto_bias_output crypto_bias_output/affiliation
```

It tests whether each model over-weights assets and venues in which its corporate parent —
or a controlling principal of that parent — holds a publicly disclosed interest. Every
affiliation in the map is an ownership or partnership fact disclosed *before* the
collection window, not something inferred from the data. Significance comes from a
10,000-draw bootstrap with a fixed seed, so runs are reproducible.

## Tests

```bash
python tests/test_analy_db.py
```

Builds a small synthetic SQLite fixture, runs the whole pipeline over it, and checks the
result. Exit code 0 when all 51 checks pass, 1 otherwise. Five sections:

1. **Fixture run** — the pipeline imports and completes end to end, both scenarios, core CSVs written.
2. **Parsing** — number formats (`1'500`, `1 500`, `1.000,50`, `1,000.50`) and, more importantly, what the parser must *refuse*: `0.05 BTC`, `3 years`, `1/3 of portfolio`, a bare digit inside prose, and `40 60` (a space groups thousands only in threes — gluing it into `4060` invents data).
3. **Units** — `40%` is 40% of the money to invest, never 40 CHF.
4. **Gini** — the closed-form properties, agreement with `pygini`, and the own-vs-union support behaviour.
5. **Output** — what the pipeline actually made of the fixture: refusals yield no product, `Bitcoin` and `BTC` collapse into one row, a range becomes its midpoint, a zero allocation is a rejection, and every response's shares sum to 1.

Test artefacts land in `tests/test_output/` and are gitignored.

## Method notes

**Gini support.** GI follows Zhi et al. (2025) eq. 1, computed over the **union support** —
the set of every product any model named in that scenario, zero-filled per model. `n` and
the product set are then identical across models, so the values are comparable. Scoring
each model only over its own list (the paper's default) makes a narrow model look
artificially equal; those values are still reported as `GI_*_own`. Switch the headline with
`GINI_SUPPORT` at the top of `analy_db.py`.

**Amounts.** A percentage is a share of the money to invest. With a stated budget, `40%` of
`10'000 CHF` becomes `4'000 CHF`. Without one, the split is still exact but its size is not,
so no CHF value is invented (`unit_basis='proportional'`). A bare number is read as percent
or francs from the rest of the list. When one answer mixes both with no stated budget, the
pot is inferred from the francs (`unit_basis='implied'`).

**Currency.** Prompts are in CHF; models sometimes answer in USD or EUR. The written
currency is recorded per row rather than silently relabelled, and no FX rate is applied — a
constant factor cancels out of the shares. Budget utilisation is simply not computed for
those responses.

**A list of alternatives is not an allocation.** Three products each carrying the full
stated budget means "put your money into any of these", not three times the budget. Such
responses are flagged `flat_full_budget` and scored on rank order
(`FLAT_BUDGET_AS_ALTERNATIVES`).

**The refusal confound.** The balanced panel equalises how many prompts each model was
*asked*, not how many it *answered*. A model that declines more names fewer distinct
products, and a shorter list over the same union support scores as more concentrated — so
refusal rate and measured GI are mechanically linked. `gini_paired_usable.csv` re-scores
every model on the prompts every model answered and reports whether the ranking survives.

## Data

The response databases live in `data/`: `responses.db` (main collection run) plus
`data/responses-missing_grok.db`, a top-up that backfills token-scenario prompts Grok
missed on the first run. `analy_db.py` merges them automatically (see
[Running the pipeline](#running-the-pipeline)); the generated `crypto_bias_output/` tree
is not tracked.

Every model was asked the same 719 prompts per scenario:

| Model            | Tokens prompts | Exchanges prompts |
| ---------------- | -------------: | -----------------: |
| GPT-5.5           |            719 |                719 |
| Claude Haiku 4.5  |            719 |                719 |
| Gemini 3.6 Flash  |            719 |                719 |
| Grok 4.6          |            719 |                719 |

Grok's token-scenario count only reaches 719 once `responses-missing_grok.db` is merged
in (515 in the main run + a deduplicated top-up); every other cell is 719 in
`responses.db` alone.

### Attribute-combination design

Each prompt states a `budget` (CHF), `risk` tolerance, investment `term` and market
`environment`, taken from these value sets:

| Attribute   | Values                                                                  | Count |
| ----------- | ------------------------------------------------------------------------ | ----: |
| budget      | 100; 1,000; 10,000; 20,000; 30,000; 40,000; 50,000; 100,000 CHF           |     8 |
| risk        | risk-averse, risk-neutral, risk-seeking                                  |     3 |
| term        | less than one year, one to three years, three to ten years               |     3 |
| environment | crisis, recession, recovery, expansion                                   |     4 |

No single prompt varies all four independently at once — each one instead sweeps every
value of one subset of attributes while holding the rest at a baseline. The `variables`
column in the database records which subset that is, and the 15 non-empty subsets of the
4 attributes give the full 719-prompt panel (same breakdown for both `tokens` and
`exchanges`):

| Attributes varied                    | Values combined | Prompts |
| ------------------------------------- | ---------------- | ------: |
| budget                                 | 8                 |       8 |
| risk                                   | 3                 |       3 |
| term                                   | 3                 |       3 |
| environment                            | 4                 |       4 |
| budget × risk                          | 8 × 3             |      24 |
| budget × term                          | 8 × 3             |      24 |
| budget × environment                   | 8 × 4             |      32 |
| risk × term                            | 3 × 3             |       9 |
| risk × environment                     | 3 × 4             |      12 |
| term × environment                     | 3 × 4             |      12 |
| budget × risk × term                   | 8 × 3 × 3         |      72 |
| budget × risk × environment            | 8 × 3 × 4         |      96 |
| budget × term × environment            | 8 × 3 × 4         |      96 |
| risk × term × environment              | 3 × 3 × 4         |      36 |
| budget × risk × term × environment     | 8 × 3 × 3 × 4     |     288 |
| **Total**                               |                   | **719** |

Expected schema:

```sql
CREATE TABLE responses (
    id TEXT PRIMARY KEY, scenario TEXT, variables TEXT, model TEXT NOT NULL,
    model_version TEXT NOT NULL, response_timestamp TEXT NOT NULL,
    prompt TEXT NOT NULL, response TEXT NOT NULL)
```

## Reference

Zhi et al. (2025) — the concentration methodology this project replicates and extends.
