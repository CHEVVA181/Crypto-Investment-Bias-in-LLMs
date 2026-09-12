#!/usr/bin/env python3
# ===========================================================================
#  affiliation_analysis.py
#
#  Provider-affiliation analysis for the master project
#  "Product Bias in Large Language Model Investment Recommendations".
#
#  Question:  do the four models over-weight assets and venues in which the
#             corporate parent, or a controlling principal of that parent,
#             holds a disclosed economic or reputational interest?
#
#  Inputs :  crypto_bias_output/tokens/parsed_recommendations.csv
#            crypto_bias_output/exchanges/parsed_recommendations.csv
#  Outputs:  affiliation_tests.csv, affiliation_lift_ranks.csv,
#            figures/fig8_affiliation.pdf/.png
# ===========================================================================
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260911)
N_BOOT = 10_000

OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("affiliation_output")
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("crypto_bias_output")
(OUT / "figures").mkdir(parents=True, exist_ok=True)

MODELS = ["GPT-5.5", "Claude Haiku 4.5", "Gemini 3.6 Flash", "Grok 4.6"]
SHORT  = {"GPT-5.5": "GPT-5.5", "Claude Haiku 4.5": "Claude Haiku",
          "Gemini 3.6 Flash": "Gemini Flash", "Grok 4.6": "Grok 4.6"}

# ---------------------------------------------------------------------------
#  The affiliation map.  Every entry is an ownership or partnership fact that
#  was publicly disclosed BEFORE the collection window, not an inference from
#  the data.  `focal` is the model whose provider holds the interest.
# ---------------------------------------------------------------------------
AFFILIATIONS = [
    dict(key="grok_btc",      scenario="tokens",    code="BTC",
         focal="Grok 4.6",
         tie="Tesla and SpaceX (Musk-controlled) hold disclosed bitcoin treasuries"),
    dict(key="grok_doge",     scenario="tokens",    code="DOGE",
         focal="Grok 4.6",
         tie="Musk's sustained public association with Dogecoin; DOGE accepted by Tesla"),
    dict(key="gemini_sol",    scenario="tokens",    code="SOL",
         focal="Gemini 3.6 Flash",
         tie="Google Cloud runs Solana validators / block-data and dev partnerships"),
    dict(key="gemini_cb",     scenario="exchanges", code="COINBASE",
         focal="Gemini 3.6 Flash",
         tie="Coinbase-Google Cloud commercial partnership"),
    dict(key="gemini_gemini", scenario="exchanges", code="GEMINI",
         focal="Gemini 3.6 Flash",
         tie="Name collision only: the Gemini exchange is unrelated to Google (placebo)"),
    dict(key="claude_ftx",    scenario="exchanges", code="FTX",
         focal="Claude Haiku 4.5",
         tie="FTX/Alameda was a major early outside shareholder in Anthropic"),
    dict(key="gpt_wld",       scenario="tokens",    code="WLD",
         focal="GPT-5.5",
         tie="OpenAI's CEO co-founded World / Worldcoin (WLD)"),
]

# ---------------------------------------------------------------------------
def load(scenario):
    df = pd.read_csv(BASE / scenario / "parsed_recommendations.csv")
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df

def response_panel(df, code):
    """One row per parsed response: did it name `code`, and what share of the
    budget did it give it?  Absences are true zeros, not missing data."""
    resp = df[["response_uid", "model"]].drop_duplicates()
    hit = (df[df["code"] == code]
             .groupby(["response_uid", "model"], as_index=False)["share"].sum())
    out = resp.merge(hit, on=["response_uid", "model"], how="left")
    out["share"] = out["share"].fillna(0.0)
    out["named"] = (out["share"] > 0).astype(int)
    # a product can be named without a parseable amount -> recover the flag
    named_any = set(df.loc[df["code"] == code, "response_uid"])
    out.loc[out["response_uid"].isin(named_any), "named"] = 1
    return out

def cluster_boot(focal_v, other_v, n=N_BOOT):
    """Non-parametric bootstrap over responses; returns the 95% CI of the
    difference in means and a two-sided bootstrap p-value."""
    d0 = focal_v.mean() - other_v.mean()
    fi = RNG.integers(0, len(focal_v), size=(n, len(focal_v)))
    oi = RNG.integers(0, len(other_v), size=(n, len(other_v)))
    d = focal_v[fi].mean(axis=1) - other_v[oi].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return d0, lo, hi, max(p, 1.0 / n)

def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        run = max(run, val)
        adj[idx] = min(run, 1.0)
    return adj

# ---------------------------------------------------------------------------
print("=" * 78)
print(" PROVIDER-AFFILIATION ANALYSIS".center(78))
print("=" * 78)

data = {s: load(s) for s in ("tokens", "exchanges")}
for s, df in data.items():
    n_resp = df["response_uid"].nunique()
    print(f"  loaded {s:10s}: {len(df):6,d} parsed rows   "
          f"{n_resp:5,d} responses   {df['code'].nunique():3d} products")
print()

rows = []
for a in AFFILIATIONS:
    df = data[a["scenario"]]
    panel = response_panel(df, a["code"])
    f = panel[panel["model"] == a["focal"]]
    o = panel[panel["model"] != a["focal"]]
    fs, os_ = f["share"].to_numpy(), o["share"].to_numpy()

    if fs.sum() == 0 and os_.sum() == 0 and f["named"].sum() == 0 and o["named"].sum() == 0:
        rows.append(dict(key=a["key"], scenario=a["scenario"], code=a["code"],
                         focal=a["focal"], n_focal=len(f), n_other=len(o),
                         focal_share=0.0, other_share=0.0, lift=np.nan,
                         diff=0.0, ci_lo=0.0, ci_hi=0.0, p=1.0,
                         focal_named=0, other_named=0, tie=a["tie"],
                         status="ABSENT FROM CORPUS"))
        continue

    diff, lo, hi, p = cluster_boot(fs, os_)
    lift = (fs.mean() / os_.mean()) if os_.mean() > 0 else np.inf
    rows.append(dict(key=a["key"], scenario=a["scenario"], code=a["code"],
                     focal=a["focal"], n_focal=len(f), n_other=len(o),
                     focal_share=fs.mean(), other_share=os_.mean(), lift=lift,
                     diff=diff, ci_lo=lo, ci_hi=hi, p=p,
                     focal_named=int(f["named"].sum()),
                     other_named=int(o["named"].sum()),
                     tie=a["tie"], status="tested"))

res = pd.DataFrame(rows)
live = res["status"] == "tested"
res.loc[live, "p_holm"] = holm(res.loc[live, "p"].to_numpy())

print("-" * 78)
print(" PRE-SPECIFIED AFFILIATION TESTS")
print(" mean allocated share of budget, focal model vs. the other three pooled")
print("-" * 78)
hdr = f"{'asset':>9} {'focal model':>14} {'focal':>8} {'others':>8} {'lift':>6} {'diff':>8} {'95% CI':>18} {'p_holm':>9}"
print(hdr); print("-" * 78)
for _, r in res.iterrows():
    if r["status"] != "tested":
        print(f"{r['code']:>9} {SHORT[r['focal']]:>14}   -- {r['status']} --")
        continue
    ci = f"[{r['ci_lo']*100:+.2f},{r['ci_hi']*100:+.2f}]"
    print(f"{r['code']:>9} {SHORT[r['focal']]:>14} "
          f"{r['focal_share']*100:7.2f}% {r['other_share']*100:7.2f}% "
          f"{r['lift']:5.2f}x {r['diff']*100:+7.2f}pp {ci:>18} {r['p_holm']:9.4f}")
print("-" * 78)
print()

# ---------------------------------------------------------------------------
#  Placebo ranking: where does the affiliated asset sit among ALL assets when
#  each model is ranked by its own-vs-others over-weight?
# ---------------------------------------------------------------------------
print("-" * 78)
print(" PLACEBO RANKING  (every product, every model, ranked by over-weight)")
print("-" * 78)
rank_rows = []
for scenario, df in data.items():
    codes = sorted(df["code"].dropna().unique())
    for m in MODELS:
        recs = []
        for c in codes:
            p = response_panel(df, c)
            fm = p.loc[p["model"] == m, "share"].mean()
            om = p.loc[p["model"] != m, "share"].mean()
            recs.append((c, fm, om, fm - om))
        d = pd.DataFrame(recs, columns=["code", "focal", "other", "diff"])
        d = d.sort_values("diff", ascending=False).reset_index(drop=True)
        d["rank"] = d.index + 1
        d["model"], d["scenario"], d["n_products"] = m, scenario, len(codes)
        rank_rows.append(d)
ranks = pd.concat(rank_rows, ignore_index=True)

for _, r in res[res["status"] == "tested"].iterrows():
    q = ranks[(ranks["scenario"] == r["scenario"]) & (ranks["model"] == r["focal"])
              & (ranks["code"] == r["code"])]
    if len(q):
        q = q.iloc[0]
        print(f"  {SHORT[r['focal']]:>14} / {r['code']:<9} "
              f"rank {int(q['rank']):3d} of {int(q['n_products']):3d} "
              f"({r['scenario']} scenario)")
print("-" * 78)
print()

print(" Largest single over-weight per model (tokens):")
for m in MODELS:
    q = ranks[(ranks["scenario"] == "tokens") & (ranks["model"] == m)].iloc[0]
    print(f"   {SHORT[m]:>14}  {q['code']:<8} {q['diff']*100:+6.2f}pp "
          f"({q['focal']*100:5.2f}% vs {q['other']*100:5.2f}%)")
print()
print(" Largest single over-weight per model (exchanges):")
for m in MODELS:
    q = ranks[(ranks["scenario"] == "exchanges") & (ranks["model"] == m)].iloc[0]
    print(f"   {SHORT[m]:>14}  {q['code']:<10} {q['diff']*100:+6.2f}pp "
          f"({q['focal']*100:5.2f}% vs {q['other']*100:5.2f}%)")
print()

res.to_csv(OUT / "affiliation_tests.csv", index=False)
ranks.to_csv(OUT / "affiliation_lift_ranks.csv", index=False)
print(f"  wrote {OUT/'affiliation_tests.csv'}")
print(f"  wrote {OUT/'affiliation_lift_ranks.csv'}")
print("=" * 78)

# ===========================================================================
#  CONFOUND CONTROL -- is the effect just list length?
#  Grok names 2.90 products per response against 4.5-5.6 for the others, and a
#  shorter list mechanically raises the share of whatever sits at the top.  The
#  test below holds the number of products named in the response fixed and
#  compares models only on the strata where all four are observed.
# ===========================================================================
MIN_CELL = 10

def length_strata(scenario, code):
    df = data[scenario]
    lens = (df.groupby(["response_uid", "model"], as_index=False)["code"]
              .nunique().rename(columns={"code": "k"}))
    pan = response_panel(df, code).merge(lens, on=["response_uid", "model"])
    tab, common = [], []
    for k in sorted(pan["k"].unique()):
        sub = pan[pan["k"] == k]
        cell = {"k": int(k), "n": int(len(sub))}
        full = True
        for m in MODELS:
            v = sub.loc[sub["model"] == m, "share"]
            cell[m] = v.mean() if len(v) >= MIN_CELL else np.nan
            cell[f"n_{m}"] = int(len(v))
            full &= len(v) >= MIN_CELL
        tab.append(cell)
        if full:
            common.append(int(k))
    tab = pd.DataFrame(tab)
    sup = tab[tab["k"].isin(common)]
    w = sup["n"] / sup["n"].sum()
    std = {m: float((sup[m] * w).sum()) for m in MODELS}
    raw = {m: float(pan.loc[pan["model"] == m, "share"].mean()) for m in MODELS}
    return tab, sup, std, raw, common

print("=" * 78)
print(" CONFOUND CONTROL: LIST LENGTH".center(78))
print("=" * 78)

strat_store = {}
for code, scen in (("BTC", "tokens"), ("DOGE", "tokens"), ("SOL", "tokens")):
    tab, sup, std, raw, common = length_strata(scen, code)
    strat_store[code] = (tab, std, raw, common)
    print(f"\n  {code} -- allocated share of budget within list-length strata")
    print("  " + f"{'k':>3} " + "".join(f"{SHORT[m]:>16}" for m in MODELS) + f"{'n':>7}")
    for _, r in tab.iterrows():
        if r["n"] < 40:
            continue
        line = f"  {int(r['k']):>3} "
        for m in MODELS:
            line += (f"{r[m]*100:9.2f}% (n={int(r['n_'+m]):<3d})"
                     if not np.isnan(r[m]) else f"{'--':>16}")
        print(line + f"{int(r['n']):>7}")
    print(f"  common support: list lengths {common}")
    print(f"  {'':>18}{'raw':>10}{'length-std':>13}{'shift':>9}")
    for m in MODELS:
        print(f"  {SHORT[m]:>18}{raw[m]*100:9.2f}%{std[m]*100:12.2f}%"
              f"{(std[m]-raw[m])*100:+8.2f}pp")
    focal = {"BTC": "Grok 4.6", "DOGE": "Grok 4.6", "SOL": "Gemini 3.6 Flash"}[code]
    others = [m for m in MODELS if m != focal]
    om = float(np.mean([std[m] for m in others]))
    verdict = "SURVIVES" if std[focal] > om else "DOES NOT SURVIVE"
    print(f"  --> length-standardised lift for {SHORT[focal]}: "
          f"{(std[focal]/om if om else np.inf):.2f}x   [{verdict}]")
print()
print("=" * 78)

pd.concat([t.assign(code=c) for c, (t, *_ ) in strat_store.items()]) \
  .to_csv(OUT / "affiliation_length_strata.csv", index=False)
strat = strat_store["BTC"][0]
strat = strat[strat["n"] >= 40].reset_index(drop=True)

# ===========================================================================
#  TWO FURTHER PROBES
#   (a) sector self-interest: all four vendors are AI companies -- do their
#       models over-weight the AI & infrastructure token category?
#   (b) XRP / Ripple, an early Google Ventures portfolio company.
# ===========================================================================
print("=" * 78)
print(" SECTOR SELF-INTEREST: 'AI & Infrastructure' token category".center(78))
print("=" * 78)
tokdf = data["tokens"]
ai_codes = sorted(tokdf.loc[tokdf["category"] == "AI & Infrastructure", "code"].unique())
print(f"  category members: {', '.join(ai_codes)}")
resp = tokdf[["response_uid", "model"]].drop_duplicates()
aihit = (tokdf[tokdf["category"] == "AI & Infrastructure"]
           .groupby(["response_uid", "model"], as_index=False)["share"].sum())
aip = resp.merge(aihit, on=["response_uid", "model"], how="left").fillna({"share": 0.0})
lens = (tokdf.groupby(["response_uid", "model"], as_index=False)["code"]
          .nunique().rename(columns={"code": "k"}))
aip = aip.merge(lens, on=["response_uid", "model"])
common = [int(k) for k in sorted(aip["k"].unique())
          if all((aip[(aip["k"] == k) & (aip["model"] == m)].shape[0] >= MIN_CELL)
                 for m in MODELS)]
sup = aip[aip["k"].isin(common)]
wt = sup.groupby("k").size() / len(sup)
print(f"  {'':>18}{'raw':>10}{'length-std':>13}{'responses':>11}")
for m in MODELS:
    raw = aip.loc[aip["model"] == m, "share"].mean()
    std = sum(wt[k] * sup[(sup["k"] == k) & (sup["model"] == m)]["share"].mean()
              for k in common)
    n = int((aip[aip["model"] == m]["share"] > 0).sum())
    print(f"  {SHORT[m]:>18}{raw*100:9.2f}%{std*100:12.2f}%{n:11d}")
print(f"  common support: list lengths {common}")
print()

print("-" * 78)
print(" XRP / Ripple  (Google Ventures led a 2015 Ripple Labs round)")
print("-" * 78)
xp = response_panel(tokdf, "XRP")
for m in MODELS:
    v = xp[xp["model"] == m]
    print(f"  {SHORT[m]:>18}  named in {int(v['named'].sum()):3d} / {len(v):4d} responses"
          f"   mean allocated share {v['share'].mean()*100:5.3f}%")
print("-" * 78)
print()

# --- length-controlled bootstrap for the one surviving signal ---------------
print("-" * 78)
print(" LENGTH-CONTROLLED BOOTSTRAP: DOGE, Grok 4.6 vs. the other three")
print("-" * 78)
dg = response_panel(tokdf, "DOGE").merge(lens, on=["response_uid", "model"])
dgc = dg[dg["k"].isin(strat_store["DOGE"][3])]
f = dgc[dgc["model"] == "Grok 4.6"]["share"].to_numpy()
o = dgc[dgc["model"] != "Grok 4.6"]["share"].to_numpy()
d0, lo, hi, pv = cluster_boot(f, o)
print(f"  Grok      mean share {f.mean()*100:6.3f}%   (n = {len(f)} responses)")
print(f"  others    mean share {o.mean()*100:6.3f}%   (n = {len(o)} responses)")
print(f"  difference {d0*100:+.3f}pp   95% CI [{lo*100:+.3f}, {hi*100:+.3f}]   p = {pv:.4f}")
dgn = response_panel(tokdf, "DOGE")
print()
print("  responses naming DOGE at all:")
for m in MODELS:
    v = dgn[dgn["model"] == m]
    print(f"    {SHORT[m]:>18}  {int(v['named'].sum()):3d} / {len(v):4d}"
          f"  ({v['named'].mean()*100:5.2f}% of responses)")
print("-" * 78)
print()

# ===========================================================================
#  FIGURE 8
# ===========================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 130, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "legend.frameon": False,
})
MODEL_COLORS = {"GPT-5.5": "green", "Claude Haiku 4.5": "orange",
                "Gemini 3.6 Flash": "blue", "Grok 4.6": "red"}
SUPPORT, AGAINST, NULLC = "#1b7f3b", "#b02318", "#8a8a8a"

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7),
                         gridspec_kw={"width_ratios": [1.12, 1.0]})
fig2, axes2 = plt.subplots(1, 2, figsize=(12.6, 4.7),
                           gridspec_kw={"width_ratios": [1.0, 1.15]})

# --- panel A: affiliation lift, log scale -----------------------------------
ax = axes[0]
tested = res[res["status"] == "tested"].copy()
tested["lab"] = [f"{r['code']} — {SHORT[r['focal']]}" for _, r in tested.iterrows()]
tested = tested.iloc[::-1]
y = np.arange(len(tested))
lifts = tested["lift"].replace(np.inf, 60.0).to_numpy()
cols = [SUPPORT if (l > 1 and p < .05) else AGAINST if (l < 1 and p < .05) else NULLC
        for l, p in zip(tested["lift"], tested["p_holm"])]
ax.barh(y, lifts, color=cols, height=0.6)
ax.axvline(1.0, color="black", lw=1.0)
ax.set_xscale("log"); ax.set_xlim(0.1, 400)
ax.set_yticks(y, tested["lab"], fontsize=8.5)
for yi, r in zip(y, tested.itertuples()):
    txt = "only model to\nallocate to it" if np.isinf(r.lift) else f"{r.lift:.2f}×"
    ax.text(lifts[yi] * 1.2, yi, txt, va="center", fontsize=7.4)
ax.set_xlabel("allocated share, focal model ÷ other three  (log scale)")
ax.set_title("A. Affiliation lift, before confound control")
ax.legend(handles=[Patch(facecolor=SUPPORT, label="over-weighted (Holm $p<.05$)"),
                   Patch(facecolor=AGAINST, label="under-weighted (Holm $p<.05$)"),
                   Patch(facecolor=NULLC,  label="not distinguishable")],
          fontsize=7.4, loc="upper right")

# --- panel B: BTC share within list-length strata ---------------------------
ax = axes[1]
ks = strat["k"].to_numpy(); width = 0.2
for i, m in enumerate(MODELS):
    ax.bar(np.arange(len(ks)) + (i - 1.5) * width, strat[m] * 100, width,
           color=MODEL_COLORS[m], label=SHORT[m])
ax.set_xticks(np.arange(len(ks)), [int(k) for k in ks])
ax.set_xlabel("number of tokens named in the response")
ax.set_ylabel("allocated share of budget to BTC (%)")
ax.set_title("B. Bitcoin: the over-weight is list length")
ax.legend(fontsize=7.5, ncol=2, loc="upper right")

# --- panel C: DOGE ----------------------------------------------------------
ax = axes2[0]
doge_raw = [strat_store["DOGE"][2][m] * 100 for m in MODELS]
doge_std = [strat_store["DOGE"][1][m] * 100 for m in MODELS]
xx = np.arange(4)
ax.bar(xx - 0.19, doge_raw, 0.36, color=[MODEL_COLORS[m] for m in MODELS],
       alpha=0.45, label="raw")
ax.bar(xx + 0.19, doge_std, 0.36, color=[MODEL_COLORS[m] for m in MODELS],
       label="length-standardised")
for i, (a, b) in enumerate(zip(doge_raw, doge_std)):
    ax.text(i - 0.19, a + 0.015, f"{a:.2f}", ha="center", fontsize=7.2)
    ax.text(i + 0.19, b + 0.015, f"{b:.2f}", ha="center", fontsize=7.2,
            fontweight="bold")
ax.set_xticks(xx, [SHORT[m] for m in MODELS], fontsize=8.5)
ax.set_ylabel("allocated share of budget to DOGE (%)")
ax.set_title("A. Dogecoin: the one signal that survives")
ax.legend(handles=[Patch(facecolor="grey", alpha=0.45, label="raw"),
                   Patch(facecolor="grey", label="length-standardised")],
          fontsize=7.5, loc="upper left")

# --- panel D: placebo distribution ------------------------------------------
ax = axes2[1]
hl = {"Grok 4.6": ["BTC", "DOGE"], "Gemini 3.6 Flash": ["SOL"]}
for i, m in enumerate(MODELS):
    d = ranks[(ranks["scenario"] == "tokens") & (ranks["model"] == m)]
    jit = RNG.normal(0, 0.055, len(d))
    ax.scatter(np.full(len(d), i) + jit, d["diff"] * 100, s=13,
               color="lightgrey", edgecolor="none", zorder=2)
    for code in hl.get(m, []):
        q = d[d["code"] == code]
        ax.scatter([i], q["diff"] * 100, s=70, color=MODEL_COLORS[m],
                   edgecolor="black", linewidth=0.7, zorder=4)
        ax.annotate(f"{code} (rank {int(q['rank'].iloc[0])}/47)",
                    (i, q["diff"].iloc[0] * 100), textcoords="offset points",
                    xytext=(9, -1), fontsize=7.5, fontweight="bold")
ax.axhline(0, color="black", lw=0.9)
ax.set_xticks(range(4), [SHORT[m] for m in MODELS], fontsize=8.5)
ax.set_xlim(-0.5, 4.55)
ax.set_ylabel("over-weight vs. the other three (pp)")
ax.set_title("B. All 47 tokens, raw; affiliated assets marked")

fig.suptitle("Provider affiliation: the bitcoin result",
             fontsize=11.5, fontweight="bold", y=1.02)
fig.savefig(OUT / "figures" / "fig8_affiliation.png", dpi=220)
fig.savefig(OUT / "figures" / "fig8_affiliation.pdf")
fig2.suptitle("Dogecoin, and where the affiliated assets rank",
              fontsize=11.5, fontweight="bold", y=1.02)
fig2.savefig(OUT / "figures" / "fig9_affiliation_doge.png", dpi=220)
fig2.savefig(OUT / "figures" / "fig9_affiliation_doge.pdf")
print(f"  wrote {OUT/'figures'/'fig8_affiliation.pdf'}")
print(f"  wrote {OUT/'figures'/'fig9_affiliation_doge.pdf'}")
print("=" * 78)
