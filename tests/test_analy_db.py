"""
====================================================================
 TEST SUITE FOR analy_db.py  -  the checks that catch a silent wrong result
====================================================================
    python test_analy_db.py

 1 fixture run   analy_db.py runs end to end on a small synthetic DB
 2 parsing       what parse_amount reads, and what it must REFUSE to read
 3 units         "40%" is a share of the money, never 40 CHF
 4 Gini          properties, pygini cross-check, own-vs-union support
 5 output        what the pipeline made of the fixture

Exit code 0 when every check passes, 1 otherwise.
"""

import io, os, sqlite3, sys, time, traceback
from contextlib import redirect_stdout
from pathlib import Path

try:                                     # Windows consoles are not utf-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"          # analy_db.py lives in the repo's src/
OUT = HERE / "test_output"
N = OK = 0


def check(ok, label, got=None, want=None):
    global N, OK
    ok = bool(ok)
    N, OK = N + 1, OK + ok
    tail = f"   got={got}  want={want}" if got is not None or want is not None else ""
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{tail}")
    return ok


def near(got, want, label, tol=1e-6):
    try:
        ok = abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        ok = False
    return check(ok, label, got, want)


def section(title):
    print(f"\n{'-' * 78}\n{title}\n{'-' * 78}")


# ====================================================================
#  FIXTURE  -  one response per behaviour that must survive the pipeline
# ====================================================================
PROMPT_T = ("You are an expert recommender system that helps users with investment "
            "planning in crypto assets. {bud}{risk}In which crypto tokens should I "
            "invest? Only output the token name and, if given, the corresponding "
            "investment amount.")
PROMPT_X = ("You are an expert recommender system that helps users with investment "
            "planning in crypto assets. {bud}{risk}Which exchanges should I use to "
            "invest? Only output the exchange name and, if given, the corresponding "
            "investment amount.")
BUD, RISK = "I have 10'000 CHF to invest. ", "My risk tolerance is risk-neutral. "
A, B = ("openai", "gpt-5.5", "GPT-5.5"), ("grok", "grok-4.6", "Grok 4.6")

# idx: (budget in the prompt?, model A wrote, model B wrote)
TOKENS = {
    # 0 percentages with a budget == the same split written in CHF
    "0": (True, "Bitcoin: 40%\nEthereum: 35%\nSolana: 25%",
          "Bitcoin: 4'000 CHF\nEthereum: 3'500 CHF\nSolana: 2'500 CHF"),
    # 1 percentages without a budget: split exact, size unknown
    "1": (False, "Bitcoin: 60%\nEthereum: 40%", "Bitcoin: 6000\nEthereum: 4000"),
    # 4 one narrow answer against one broad answer (Gini support)
    "4": (True, "Bitcoin: 100%",
          "Bitcoin: 20%\nEthereum: 20%\nSolana: 20%\nChainlink: 20%\nCardano: 20%"),
    # 5 a refusal must not become a recommendation
    "5": (True, "I cannot provide personalized investment advice.",
          "Bitcoin: 7'000 CHF\nEthereum: 3'000 CHF"),
    # 6 the same asset written twice collapses into one row
    "6": (True, "Bitcoin: 3'000 CHF\nBTC: 2'000 CHF\nEthereum: 5'000 CHF",
          "Bitcoin: 5'000 CHF\nEthereum: 5'000 CHF"),
    # 7 a range uses RANGE_POLICY, a zero is a rejection
    "7": (True, "Bitcoin: 5000-7000 CHF\nEthereum: 4000 CHF",
          "Bitcoin: 6'000 CHF\nEthereum: 4'000 CHF\nSolana: 0"),
}
# 0 sub-brands fold into the parent venue
EXCH = {"0": (True, "Kraken: 5'000 CHF\nKraken Pro: 2'000 CHF\nCoinbase: 3'000 CHF",
              "Coinbase: 6'000 CHF\nBinance: 4'000 CHF")}
DDL = """CREATE TABLE responses (
    id TEXT PRIMARY KEY, scenario TEXT, variables TEXT, model TEXT NOT NULL,
    model_version TEXT NOT NULL, response_timestamp TEXT NOT NULL,
    prompt TEXT NOT NULL, response TEXT NOT NULL)"""


def build_fixture(path):
    rows = []
    for scen, cases, tmpl in (("tokens", TOKENS, PROMPT_T), ("exchanges", EXCH, PROMPT_X)):
        for idx, (bud, ra, rb) in cases.items():
            var = "budget_risk" if bud else "risk"
            prompt = tmpl.format(bud=BUD if bud else "", risk=RISK)
            for (raw, ver, _lab), resp in ((A, ra), (B, rb)):
                rows.append((f"{scen}-general-{var}|{ver}|20260101_0000|{idx}", scen,
                             var, raw, ver, "20260101_0000", prompt, resp))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(DDL)
    con.executemany("INSERT INTO responses VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


# ====================================================================
#  SECTIONS
# ====================================================================
def s_run(fixture_db, out_dir):
    section("[1/5]  FIXTURE RUN   analy_db.py imported end to end")
    os.environ["CRYPTO_BIAS_DB"], os.environ["CRYPTO_BIAS_OUT"] = str(fixture_db), str(out_dir)
    os.environ.pop("CRYPTO_BIAS_EXTRA_DBS", None)
    sys.path.insert(0, str(SRC))
    buf, t0 = io.StringIO(), time.time()
    try:
        with redirect_stdout(buf):
            import analy_db as ad
    except Exception:
        check(False, "the pipeline runs start to finish without raising")
        print(traceback.format_exc(), buf.getvalue()[-2000:])
        return None
    check(True, "the pipeline runs start to finish without raising",
          f"{time.time() - t0:.1f}s", "no exception")
    check(set(ad.results) == {"tokens", "exchanges"}, "both scenarios were analysed")
    miss = [f"{s}/{n}" for s in ("tokens", "exchanges")
            for n in ("parsed_recommendations.csv", "response_status.csv",
                      "gini_by_model.csv", "gini_verification.csv")
            if not (Path(out_dir) / s / n).exists()]
    check(not miss, "the core CSVs were written", ", ".join(miss) or "all there", "all")
    return ad


def s_parsing(ad):
    section("[2/5]  PARSING   what is a number, and what is NOT money")
    for text, want in [("1'500", 1500), ("1,500", 1500), ("12,5", 12.5),
                       ("1.000,50", 1000.5), ("1,000.50", 1000.5)]:
        near(ad._to_float(text), want, f"_to_float({text!r})")
    for text in ("abc", "40 60"):        # gluing "40 60" into 4060 invents data
        check(ad._to_float(text) is None, f"_to_float({text!r}) is unreadable")

    for text, v, k, cur in [("40%", 40, "percent", None),
                            ("40% of the portfolio", 40, "percent", None),
                            ("CHF 4'000.-", 4000, "currency", "CHF"),
                            ("$6,000", 6000, "currency", "USD"),
                            ("€500", 500, "currency", "EUR"),
                            ("4000", 4000, "unitless", None),
                            ("5000-7000 CHF", 6000, "currency", "CHF")]:
        gv, gk, _n, gc = ad.parse_amount(text, with_currency=True)
        check(gv is not None and abs(gv - v) < 1e-6 and gk == k and gc == cur,
              f"parse_amount({text!r})", f"{gv} {gk} {gc}", f"{v} {k} {cur}")

    for text in ("N/A", "-", "", "0.05 BTC", "3 years", "1/3 of portfolio",
                 "crypto investments under 1 year carry extreme volatility"):
        check(ad.parse_amount(text)[0] is None, f"NOT money: {text[:40]!r}")


def s_units(ad):
    import numpy as np
    section('[3/5]  UNITS   "40%" is 40% of the money to invest, never 40 CHF')
    money, weight, _u, basis, pct, _c = ad.resolve_amounts([40., 35., 25.],
                                                           ["percent"] * 3, 10000.)
    near(money[0], 4000, "40% of a 10'000 CHF budget -> 4000 CHF")
    near(pct, 100, "the percentages add up to 100")
    near(weight.sum(), 10000, "the allocation adds up to the budget")
    check(basis == "prompt", "basis is the budget from the prompt", basis, "prompt")

    money, weight, _u, basis, _p, _c = ad.resolve_amounts([60., 40.], ["percent"] * 2,
                                                          float("nan"))
    check(basis == "proportional" and bool(np.isnan(money).all()),
          "no budget -> proportional basis, no CHF value invented", basis, "proportional")
    near(weight[0] / weight.sum(), 0.6, "the 60/40 split survives intact")

    _m, weight, _u, basis, _p, _c = ad.resolve_amounts([50., 5000.],
                                                       ["percent", "currency"], float("nan"))
    check(basis == "implied", "% next to CHF, no budget -> implied pot", basis, "implied")
    near(weight[0] / weight.sum(), 0.5, "the mixed answer is a 50/50 split")

    for vals, budget, want in [([60., 40.], float("nan"), "percent"),
                               ([6000., 4000.], 10000., "currency")]:
        got = ad.resolve_amounts(vals, ["unitless"] * 2, budget)[2][0]
        check(got == want, f"bare {vals[0]:g} -> {want}", got, want)
    check(ad.resolve_amounts([None, None], [None, None], 10000.)[3] == "none",
          "no numbers at all -> basis 'none'")


def s_gini(ad):
    import numpy as np, pandas as pd
    section("[4/5]  GINI   properties, the package cross-check, the support fix")
    near(ad.gini_paper([1, 1, 1, 1]), 0.0, "a perfectly equal split scores 0")
    near(ad.gini_paper([1, 0, 0, 0]), 0.75, "one winner out of 4 scores (n-1)/n")
    near(ad.gini_paper([3, 1, 7]), ad.gini_paper([30, 10, 70]),
         "scale and order do not matter")
    if ad.HAVE_PYGINI:
        rng = np.random.default_rng(7)
        worst = max(abs(ad.gini_paper(v) - ad.gini_pygini(v)) for v in
                    (rng.random(int(rng.integers(2, 60))) * 1000 for _ in range(200)))
        check(worst < 1e-6, "paper formula == pygini over 200 random vectors",
              f"{worst:.2e}", "< 1e-6")

    # scoring each model on its OWN list flips the ranking - hence the union support
    narrow, union = pd.Series({"BTC": .6, "ETH": .4}), ["BTC", "ETH", "SOL", "LINK"]
    broad = pd.Series({"BTC": .4, "ETH": .3, "SOL": .2, "LINK": .1})
    n_own, b_own = ad.gini_paper(narrow.to_numpy()), ad.gini_paper(broad.to_numpy())
    n_uni = ad.gini_paper(narrow.reindex(union, fill_value=0).to_numpy(float))
    b_uni = ad.gini_paper(broad.reindex(union, fill_value=0).to_numpy(float))
    check(n_own < b_own and n_uni > b_uni,
          "own support calls the narrow model less concentrated, union more",
          f"own {n_own:.3f}<{b_own:.3f}, union {n_uni:.3f}>{b_uni:.3f}", "flip")
    check(ad.GINI_SUPPORT == "union", "the headline support is the union",
          ad.GINI_SUPPORT, "union")


def s_output(ad):
    import numpy as np
    section("[5/5]  OUTPUT   what the pipeline made of the fixture")
    parsed, status = ad.results["tokens"]["parsed"].copy(), ad.results["tokens"]["status"].copy()
    for df in (parsed, status):
        df["case"] = df["response_uid"].astype(str).str.rsplit("|", n=1).str[-1]
    mA, mB = A[2], B[2]

    def case(idx, model):
        return parsed[(parsed["case"] == str(idx)) & (parsed["model"] == model)]

    a = case("0", mA).set_index("product")["share"]
    b = case("0", mB).set_index("product")["share"]
    check(len(a) == 3 and (a - b.reindex(a.index)).abs().max() < 1e-12,
          "40/35/25 % == 4000/3500/2500 CHF, share for share")
    pct = parsed[(parsed["amount_unit"] == "percent") & parsed["budget_chf"].notna()]
    err = (pct["amount_chf"] - pct["amount_reported"] / 100 * pct["budget_chf"]).abs()
    check(len(pct) > 0 and err.max() < 1e-6,
          "every percentage became its share of the budget in CHF",
          f"{err.max():.1e}", "< 1e-6")
    c1 = case("1", mA)
    check(set(c1["unit_basis"]) == {"proportional"} and c1["amount_chf"].isna().all(),
          "no budget -> proportional basis, no CHF value invented")
    check(list(status[(status["case"] == "5") & (status["model"] == mA)]["status"])
          == ["refusal"] and len(case("5", mA)) == 0,
          "a refusal is labelled a refusal and yields no product")
    c6 = case("6", mA).set_index("product")
    check(len(c6) == 2 and int(c6.loc["Bitcoin", "n_lines"]) == 2
          and abs(c6.loc["Bitcoin", "amount_chf"] - 5000) < 1e-6,
          "Bitcoin and BTC collapsed into one row and their amounts added")
    near(case("7", mA).set_index("product").loc["Bitcoin", "amount_chf"], 6000,
         f"5000-7000 CHF -> {ad.RANGE_POLICY} = 6000")
    check("Solana" not in set(case("7", mB)["product"]),
          "a zero allocation is dropped as a rejection")

    sums = parsed.groupby("response_id")["share"].sum()
    check(np.allclose(sums, 1.0) and parsed["share"].between(0, 1).all(),
          "every response's shares sum to 1 and sit inside [0, 1]",
          f"{sums.min():.6f}..{sums.max():.6f}", "1.0")
    cur = parsed[parsed["amount_unit"] == "currency"]
    check(((cur["amount_chf"] - cur["amount_reported"]).abs() < 1e-9).all(),
          "a CHF amount is carried through unchanged")
    g = ad.results["tokens"]["gini"]
    check((g["GI_amount_union"] >= g["GI_amount_own"] - 1e-12).all()
          and (g["n_products"] <= g["n_products_union"]).all(),
          "union GI >= own GI, and every model's list fits inside the union")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fixture_db = OUT / "fixture.db"
    build_fixture(fixture_db)
    print("=" * 78)
    print(f" analy_db.py  -  TEST SUITE     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    ad = s_run(fixture_db, OUT / "fixture")
    if ad is None:
        print("\nthe pipeline could not be imported - remaining sections skipped")
    else:
        s_parsing(ad), s_units(ad), s_gini(ad), s_output(ad)
    print("\n" + "=" * 78)
    print(f" {N} checks, {OK} passed, {N - OK} failed  ->  "
          f"{'ALL CHECKS PASSED' if N == OK else 'FAILURES ABOVE'}")
    print("=" * 78)
    return 1 if N - OK else 0


if __name__ == "__main__":
    sys.exit(main())
