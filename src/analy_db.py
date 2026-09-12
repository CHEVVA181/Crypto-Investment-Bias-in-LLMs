import os
import re
import sys
import sqlite3
import warnings
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

# Response database. Override with the CRYPTO_BIAS_DB environment variable;
# the default is <repo>/data/responses.db.
REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("CRYPTO_BIAS_DB",
                              str(REPO_ROOT / "data" / "responses.db")))

_extra_env = os.environ.get("CRYPTO_BIAS_EXTRA_DBS")
if _extra_env is None:
    EXTRA_DB_PATHS = sorted(DB_PATH.parent.glob("responses-missing*.db"))
elif _extra_env.strip().lower() in {"", "none", "0", "off"}:
    EXTRA_DB_PATHS = []
else:
    EXTRA_DB_PATHS = [Path(x) for x in _extra_env.split(os.pathsep) if x.strip()]
EXTRA_DB_PATHS = [p for p in EXTRA_DB_PATHS if p.resolve() != DB_PATH.resolve()]

DUPLICATE_POLICY = "primary"

OUT_ROOT = Path(os.environ.get("CRYPTO_BIAS_OUT",
                               str(DB_PATH.parent / "crypto_bias_output")))

ZERO_AMOUNT_AS_REJECTION = True
RANK_WEIGHTING = "linear"
RANGE_POLICY = "midpoint"

GINI_SUPPORT = "union"

UNITLESS_AS_PERCENT_MAX = 100.0
PCT_SUM_TOL = 15.0

LOOSE_MAX_WORDS_WITHOUT_UNIT = 6

FLAT_BUDGET_AS_ALTERNATIVES = True

SAVE_PDF = True
SCENARIOS = ("tokens", "exchanges")

def read_responses(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"response DB not found: {path}")
    con = sqlite3.connect(str(path))
    try:
        frame = pd.read_sql("SELECT * FROM responses", con)
    finally:
        con.close()
    frame["source_db"] = Path(path).name
    return frame

def split_ids(frame: pd.DataFrame) -> pd.DataFrame:
    parts = frame["id"].astype(str).str.split("|", expand=True)
    frame["condition"] = parts[0]
    frame["model_raw"] = parts[1]
    frame["run_stamp"] = parts[2]
    frame["prompt_idx"] = parts[3]
    frame["prompt_key"] = frame["condition"] + "#" + frame["prompt_idx"]
    return frame

CELL_KEY = ["scenario", "condition", "model_raw", "prompt_idx"]

_sources = [DB_PATH] + list(EXTRA_DB_PATHS)
_frames = [split_ids(read_responses(p)) for p in _sources]
db = pd.concat(_frames, ignore_index=True)

_n_before = len(db)
if DUPLICATE_POLICY == "latest":
    db = db.sort_values("run_stamp", kind="mergesort")
    db = db.drop_duplicates(subset=CELL_KEY, keep="last")
else:
    db = db.drop_duplicates(subset=CELL_KEY, keep="first")
db = db.sort_index().reset_index(drop=True)
_n_dupes = _n_before - len(db)

_merge_log = (db.groupby(["source_db", "model_raw", "scenario"])
                .size().rename("responses").reset_index())

MODEL_LABELS = {
    "openai:gpt-5.5": "GPT-5.5",
    "gpt-5.5": "GPT-5.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "grok-4.6": "Grok 4.6",
}
db["model"] = db["model_raw"].map(MODEL_LABELS).fillna(db["model_raw"])
db["response"] = db["response"].fillna("")

ATTRIBUTES = ["budget", "risk", "term", "environment"]
db["attrs"] = db["variables"].fillna("")
for a in ATTRIBUTES:
    db[f"has_{a}"] = db["attrs"].str.contains(a, regex=False)

BUDGET_RE = re.compile(r"I have ([\d'’,\.]+)\s*CHF to invest", re.I)
RISK_RE = re.compile(r"My risk tolerance is ([a-z\- ]+?)\.", re.I)
TERM_RE = re.compile(r"My investment term is ([a-z\- ]+?)\.", re.I)
ENV_RE = re.compile(r"The market environment is ([a-z\- ]+?)\.", re.I)

def _budget_num(s):
    if s is None:
        return np.nan
    return float(re.sub(r"[^\d]", "", s))

db["budget_chf"] = db["prompt"].str.extract(BUDGET_RE)[0].map(
    lambda s: _budget_num(s) if isinstance(s, str) else np.nan)
db["risk_value"] = db["prompt"].str.extract(RISK_RE)[0].str.strip().str.lower()
db["term_value"] = db["prompt"].str.extract(TERM_RE)[0].str.strip().str.lower()
db["env_value"] = db["prompt"].str.extract(ENV_RE)[0].str.strip().str.lower()

BUDGET_ORDER = [100, 1000, 10000, 20000, 30000, 40000, 50000, 100000]
RISK_ORDER = ["risk-averse", "risk-neutral", "risk-seeking"]
TERM_ORDER = ["less than one year", "one to three years", "three to ten years"]
ENV_ORDER = ["crisis", "recession", "recovery", "expansion"]

MODEL_ORDER = [m for m in ["GPT-5.5", "Claude Haiku 4.5", "Gemini 3.6 Flash", "Grok 4.6"]
               if m in set(db["model"])]
MODEL_ORDER += [m for m in sorted(db["model"].unique().tolist()) if m not in MODEL_ORDER]

print("=" * 74)
print(f"DB: {DB_PATH}")
if EXTRA_DB_PATHS:
    for _p in EXTRA_DB_PATHS:
        _added = int((db["source_db"] == _p.name).sum())
        print(f"  + top-up: {_p.name}  ({_added:,} new responses kept)")
    print(f"  duplicate policy: {DUPLICATE_POLICY}  "
          f"({_n_dupes:,} duplicate cells dropped of {_n_before:,} rows read)")
else:
    print("  + top-up: none (CRYPTO_BIAS_EXTRA_DBS disabled or no files found)")
print(f"{len(db):,} responses | scenarios: {sorted(db['scenario'].dropna().unique().tolist())}")
print(pd.crosstab(db["model"], db["scenario"]).reindex(MODEL_ORDER).to_string())
print("attribute value coverage:")
print(f"  budget      {int(db['budget_chf'].notna().sum()):>5,}  values {sorted(db['budget_chf'].dropna().astype(int).unique().tolist())}")
print(f"  risk        {int(db['risk_value'].notna().sum()):>5,}  values {sorted(db['risk_value'].dropna().unique().tolist())}")
print(f"  term        {int(db['term_value'].notna().sum()):>5,}  values {sorted(db['term_value'].dropna().unique().tolist())}")
print(f"  environment {int(db['env_value'].notna().sum()):>5,}  values {sorted(db['env_value'].dropna().unique().tolist())}")

_cov = (db.pivot_table(index="prompt_key", columns="model", values="id",
                       aggfunc="size").notna())
_missing = (~_cov).sum().sort_values(ascending=False)
_missing = _missing[_missing > 0]
if len(_missing):
    print("prompt keys still missing per model (these prompts are dropped for all models):")
    for _m, _n in _missing.items():
        print(f"  {_m:<20} {int(_n):>5,}")
else:
    print("prompt coverage: complete - every model answered every prompt key")

TOKEN_REGISTRY = {
    "Bitcoin":            ("BTC",    ["bitcoin", "btc", "bitcoin (btc)", "xbt"]),
    "Ethereum":           ("ETH",    ["ethereum", "eth", "ether", "ethereum (eth)", "ether (eth)"]),
    "Solana":             ("SOL",    ["solana", "sol", "solana (sol)"]),
    "Chainlink":          ("LINK",   ["chainlink", "link", "chainlink (link)"]),
    "USD Coin":           ("USDC",   ["usd coin", "usdc", "usd coin (usdc)", "circle usdc"]),
    "Tether":             ("USDT",   ["tether", "usdt", "tether (usdt)"]),
    "Dai":                ("DAI",    ["dai", "dai (dai)", "makerdao dai"]),
    "Render":             ("RENDER", ["render", "render token", "rndr", "render (render)", "render (rndr)"]),
    "Avalanche":          ("AVAX",   ["avalanche", "avax", "avalanche (avax)"]),
    "Sui":                ("SUI",    ["sui", "sui (sui)"]),
    "Polkadot":           ("DOT",    ["polkadot", "dot", "polkadot (dot)"]),
    "NEAR Protocol":      ("NEAR",   ["near protocol", "near", "near protocol (near)", "near (near)"]),
    "Polygon":            ("POL",    ["polygon", "matic", "pol", "polygon (pol)", "polygon (matic)"]),
    "Bittensor":          ("TAO",    ["bittensor", "tao", "bittensor (tao)"]),
    "Cardano":            ("ADA",    ["cardano", "ada", "cardano (ada)"]),
    "Aave":               ("AAVE",   ["aave", "aave (aave)"]),
    "PAX Gold":           ("PAXG",   ["pax gold", "paxg", "pax gold (paxg)", "paxos gold", "paxos gold (paxg)"]),
    "Uniswap":            ("UNI",    ["uniswap", "uni", "uniswap (uni)"]),
    "Arbitrum":           ("ARB",    ["arbitrum", "arb", "arbitrum (arb)"]),
    "Pepe":               ("PEPE",   ["pepe", "pepe (pepe)", "pepe coin"]),
    "dogwifhat":          ("WIF",    ["dogwifhat", "wif", "dogwifhat (wif)", "dog wif hat"]),
    "Dogecoin":           ("DOGE",   ["dogecoin", "doge", "dogecoin (doge)"]),
    "Injective":          ("INJ",    ["injective", "inj", "injective (inj)"]),
    "Fetch.ai":           ("FET",    ["fetch.ai", "fet", "fetch ai", "fetch.ai (fet)",
                                      "artificial superintelligence alliance", "asi"]),
    "Pendle":             ("PENDLE", ["pendle", "pendle (pendle)"]),
    "Celestia":           ("TIA",    ["celestia", "tia", "celestia (tia)"]),
    "BNB":                ("BNB",    ["bnb", "binance coin", "bnb (bnb)"]),
    "Ondo Finance":       ("ONDO",   ["ondo", "ondo finance", "ondo (ondo)"]),
    "Optimism":           ("OP",     ["optimism", "op", "optimism (op)"]),
    "Bonk":               ("BONK",   ["bonk", "bonk (bonk)"]),
    "XRP":                ("XRP",    ["xrp", "ripple", "ripple (xrp)"]),
    "Monero":             ("XMR",    ["monero", "xmr", "monero (xmr)"]),
    "Kaspa":              ("KAS",    ["kaspa", "kas", "kaspa (kas)"]),
    "Aptos":              ("APT",    ["aptos", "apt", "aptos (apt)"]),
    "Shiba Inu":          ("SHIB",   ["shiba inu", "shib", "shiba inu (shib)"]),
    "Floki":              ("FLOKI",  ["floki", "floki inu", "floki (floki)"]),
    "Tether Gold":        ("XAUT",   ["tether gold", "xaut", "tether gold (xaut)"]),
    "CryptoFranc":        ("XCHF",   ["cryptofranc", "xchf", "cryptofranc (xchf)", "crypto franc"]),
    "VNX Swiss Franc":    ("VCHF",   ["vnx swiss franc", "vchf", "vnx chf", "vnx swiss franc (vchf)"]),
    "Litecoin":           ("LTC",    ["litecoin", "ltc", "litecoin (ltc)"]),
    "Cosmos":             ("ATOM",   ["cosmos", "atom", "cosmos (atom)"]),
    "Toncoin":            ("TON",    ["toncoin", "ton", "toncoin (ton)"]),
    "Hyperliquid":        ("HYPE",   ["hyperliquid", "hype", "hyperliquid (hype)"]),
    "Lido DAO":           ("LDO",    ["lido dao", "ldo", "lido", "lido dao (ldo)"]),
    "Worldcoin":          ("WLD",    ["worldcoin", "wld", "worldcoin (wld)"]),
    "Stellar":            ("XLM",    ["stellar", "xlm", "stellar (xlm)"]),
    "Fantom":             ("FTM",    ["fantom", "ftm", "fantom (ftm)", "sonic"]),
    "Immutable":          ("IMX",    ["immutable", "imx", "immutable x", "immutable (imx)"]),
    "Filecoin":           ("FIL",    ["filecoin", "fil", "filecoin (fil)"]),
    "Sei":                ("SEI",    ["sei", "sei (sei)"]),
    "Ethena":             ("ENA",    ["ethena", "ena", "ethena (ena)"]),
    "Maker":              ("MKR",    ["maker", "mkr", "makerdao", "maker (mkr)", "sky"]),
    "Jupiter":            ("JUP",    ["jupiter", "jup", "jupiter (jup)"]),
    "PayPal USD":         ("PYUSD",  ["paypal usd", "pyusd", "paypal usd (pyusd)"]),
    "Euro Coin":          ("EURC",   ["euro coin", "eurc", "euro coin (eurc)", "eur coin",
                                      "eur coin (eurc)", "circle euro coin"]),

    "Stablecoin (unspecified)": ("STABLE", [
        "stablecoin", "stablecoins", "stablecoin (unspecified)", "stable",
        "stablecoin (usdc/usdt)", "stablecoin (usdc or usdt)",
        "stablecoin (usdc)", "stablecoin (usdt)", "usdc/usdt", "stablecoin basket"]),
}

EXCHANGE_REGISTRY = {
    "Kraken":         ("KRAKEN", ["kraken", "kraken pro", "kraken futures", "kraken staking",
                                  "kraken (staking)", "kraken exchange", "kraken.com",
                                  "kraken futures (avoid)", "kraken (futures)",
                                  "kraken (for derivatives only, minimal allocation)"]),
    "Coinbase":       ("COINBASE", ["coinbase", "coinbase advanced", "coinbase pro",
                                    "coinbase exchange", "coinbase advanced trade",
                                    "coinbase one", "coinbase.com"]),
    "Bitstamp":       ("BITSTAMP", ["bitstamp", "bitstamp.net"]),
    "Binance":        ("BINANCE", ["binance", "binance.com", "binance global",
                                   "binance international"]),
    "Binance.US":     ("BINANCEUS", ["binance.us", "binance us", "binance-us"]),
    "Gemini":         ("GEMINI", ["gemini", "gemini exchange", "gemini activetrader",
                                  "gemini.com"]),
    "Swissquote":     ("SWISSQUOTE", ["swissquote", "swissquote bank", "swissquote crypto"]),
    "Bybit":          ("BYBIT", ["bybit", "bybit.com"]),
    "SwissBorg":      ("SWISSBORG", ["swissborg", "swiss borg", "swissborg app"]),
    "OKX":            ("OKX", ["okx", "okex", "okx.com"]),
    "Bitpanda":       ("BITPANDA", ["bitpanda", "bitpanda pro"]),
    "Bitcoin Suisse": ("BTCSUISSE", ["bitcoin suisse", "bitcoinsuisse", "bitcoin suisse ag"]),
    "KuCoin":         ("KUCOIN", ["kucoin", "ku coin"]),
    "MEXC":           ("MEXC", ["mexc", "mexc global"]),
    "Gate.io":        ("GATEIO", ["gate.io", "gate io", "gateio", "gate"]),
    "Crypto.com":     ("CRYPTOCOM", ["crypto.com", "crypto com", "cryptocom",
                                     "crypto.com exchange"]),
    "Sygnum":         ("SYGNUM", ["sygnum", "sygnum bank", "sygnum bank ag"]),
    "Bitget":         ("BITGET", ["bitget"]),
    "Hyperliquid":    ("HYPERLIQUID", ["hyperliquid"]),
    "Deribit":        ("DERIBIT", ["deribit"]),
    "Nexo":           ("NEXO", ["nexo"]),
    "Luno":           ("LUNO", ["luno"]),
    "Upbit":          ("UPBIT", ["upbit"]),
    "Uphold":         ("UPHOLD", ["uphold"]),
    "Bull Bitcoin":   ("BULLBTC", ["bullbitcoin", "bull bitcoin"]),
    "HTX":            ("HTX", ["htx", "huobi", "huobi global"]),
    "Bitvavo":        ("BITVAVO", ["bitvavo"]),
    "dYdX":           ("DYDX", ["dydx", "dy/dx"]),
    "Uniswap (DEX)":  ("UNISWAP", ["uniswap", "uniswap dex"]),
    "Fidelity Crypto": ("FIDELITY", ["fidelity crypto", "fidelity digital assets", "fidelity"]),
    "FTX":            ("FTX", ["ftx", "ftx.com", "ftx*"]),
}

TOKEN_CATEGORY = {
    "Bitcoin": "Store of Value (PoW)", "Kaspa": "Store of Value (PoW)",
    "Monero": "Store of Value (PoW)",
    "Ethereum": "Smart-Contract L1", "Solana": "Smart-Contract L1",
    "Cardano": "Smart-Contract L1", "Avalanche": "Smart-Contract L1",
    "Polkadot": "Smart-Contract L1", "NEAR Protocol": "Smart-Contract L1",
    "Sui": "Smart-Contract L1", "Aptos": "Smart-Contract L1",
    "Toncoin": "Smart-Contract L1", "Cosmos": "Smart-Contract L1",
    "Sei": "Smart-Contract L1", "Fantom": "Smart-Contract L1",
    "Injective": "Smart-Contract L1",
    "Polygon": "Layer 2 & Scaling", "Arbitrum": "Layer 2 & Scaling",
    "Optimism": "Layer 2 & Scaling", "Immutable": "Layer 2 & Scaling",
    "Uniswap": "DeFi", "Aave": "DeFi", "Lido DAO": "DeFi",
    "Pendle": "DeFi", "Hyperliquid": "DeFi", "Ethena": "DeFi",
    "Maker": "DeFi", "Jupiter": "DeFi",
    "Chainlink": "AI & Infrastructure", "Render": "AI & Infrastructure",
    "Bittensor": "AI & Infrastructure", "Fetch.ai": "AI & Infrastructure",
    "Filecoin": "AI & Infrastructure", "Celestia": "AI & Infrastructure",
    "Worldcoin": "AI & Infrastructure",
    "XRP": "Payments", "Stellar": "Payments", "Litecoin": "Payments",
    "BNB": "Exchange & Platform",
    "USD Coin": "Stablecoin", "Tether": "Stablecoin", "Dai": "Stablecoin",
    "CryptoFranc": "Stablecoin", "Stablecoin (unspecified)": "Stablecoin",
    "PayPal USD": "Stablecoin", "Euro Coin": "Stablecoin",
    "VNX Swiss Franc": "Stablecoin",
    "PAX Gold": "RWA & Tokenized", "Tether Gold": "RWA & Tokenized",
    "Ondo Finance": "RWA & Tokenized",
    "Dogecoin": "Meme", "Shiba Inu": "Meme", "Pepe": "Meme",
    "Bonk": "Meme", "dogwifhat": "Meme", "Floki": "Meme",
}

TOKEN_TIER_BY_CATEGORY = {
    "Stablecoin": "Stable", "RWA & Tokenized": "Stable",
    "Store of Value (PoW)": "Blue chip", "Smart-Contract L1": "Large-cap alt",
    "Exchange & Platform": "Large-cap alt", "Payments": "Large-cap alt",
    "Layer 2 & Scaling": "Small-cap alt", "DeFi": "Small-cap alt",
    "AI & Infrastructure": "Small-cap alt", "Meme": "Meme",
}
BLUE_CHIPS = {"Bitcoin", "Ethereum"}

TOKEN_CATEGORY_ORDER = ["Store of Value (PoW)", "Smart-Contract L1", "Layer 2 & Scaling",
                        "DeFi", "AI & Infrastructure", "Payments", "Exchange & Platform",
                        "RWA & Tokenized", "Stablecoin", "Meme"]
TOKEN_TIER_ORDER = ["Stable", "Blue chip", "Large-cap alt", "Small-cap alt", "Meme"]

EXCHANGE_CATEGORY = {
    "Coinbase": "US-listed CEX", "Kraken": "US-listed CEX",
    "Gemini": "US-listed CEX", "Binance.US": "US-listed CEX",
    "Fidelity Crypto": "US-listed CEX",
    "Binance": "Global CEX", "Bybit": "Global CEX", "OKX": "Global CEX",
    "KuCoin": "Global CEX", "MEXC": "Global CEX", "Gate.io": "Global CEX",
    "Bitget": "Global CEX", "Crypto.com": "Global CEX", "HTX": "Global CEX",
    "Upbit": "Global CEX",
    "Bitstamp": "EU-regulated CEX", "Bitpanda": "EU-regulated CEX",
    "Bitvavo": "EU-regulated CEX", "Luno": "EU-regulated CEX",
    "Uphold": "EU-regulated CEX",
    "Swissquote": "Swiss bank/broker", "SwissBorg": "Swiss bank/broker",
    "Bitcoin Suisse": "Swiss bank/broker", "Sygnum": "Swiss bank/broker",
    "Bull Bitcoin": "Non-custodial/brokerage", "Nexo": "CeFi lending",
    "Deribit": "Derivatives venue",
    "Hyperliquid": "On-chain / DEX", "dYdX": "On-chain / DEX",
    "Uniswap (DEX)": "On-chain / DEX",
    "FTX": "Defunct",
}
EXCHANGE_CATEGORY_ORDER = ["US-listed CEX", "EU-regulated CEX", "Swiss bank/broker",
                           "Global CEX", "Derivatives venue", "CeFi lending",
                           "Non-custodial/brokerage", "On-chain / DEX", "Defunct"]

EXCHANGE_TIER_BY_CATEGORY = {
    "Swiss bank/broker": "Swiss regulated",
    "EU-regulated CEX": "EU regulated",
    "US-listed CEX": "US regulated",
    "Global CEX": "Global / offshore",
    "Derivatives venue": "Global / offshore",
    "CeFi lending": "Global / offshore",
    "Non-custodial/brokerage": "Non-custodial",
    "On-chain / DEX": "Non-custodial",
    "Defunct": "Defunct / failed",
}
EXCHANGE_TIER_ORDER = ["Swiss regulated", "EU regulated", "US regulated",
                       "Global / offshore", "Non-custodial", "Defunct / failed"]

SCENARIO_CFG = {
    "tokens": dict(
        registry=TOKEN_REGISTRY, category=TOKEN_CATEGORY,
        category_order=TOKEN_CATEGORY_ORDER,
        tier_by_category=TOKEN_TIER_BY_CATEGORY, tier_order=TOKEN_TIER_ORDER,
        product_word="token", cat_word="category", tier_word="risk tier",
    ),
    "exchanges": dict(
        registry=EXCHANGE_REGISTRY, category=EXCHANGE_CATEGORY,
        category_order=EXCHANGE_CATEGORY_ORDER,
        tier_by_category=EXCHANGE_TIER_BY_CATEGORY, tier_order=EXCHANGE_TIER_ORDER,
        product_word="exchange", cat_word="venue type", tier_word="regulatory tier",
    ),
}

for scen, cfg in SCENARIO_CFG.items():
    missing = set(cfg["registry"]) - set(cfg["category"])
    if missing:
        raise KeyError(f"[{scen}] products without a category: {sorted(missing)}")
    assert set(cfg["category"].values()) <= set(cfg["category_order"]), scen

NON_PRODUCT_PATTERNS = [
    r"^\d+\.\s*(your|consult|conduct|understand|consider)",
    r"risk (tolerance|assessment|profile|management|level)",
    r"financial (goals|advisor|situation|assessment|outcomes)",
    r"investment (timeline|horizon|goals|strategy|experience|amount|recommendation)",
    r"portfolio (size|composition)", r"existing (holdings|portfolio)",
    r"liquidity needs", r"emergency fund", r"tax (consideration|situation)",
    r"geographic (location|restrictions)", r"time availability",
    r"what (i|you) (recommend|should|can)", r"instead", r"^important", r"^note",
    r"^disclaimer", r"^none$", r"licensed", r"whitepaper", r"on-chain analytics",
    r"fundamentals analysis", r"^based on", r"asset allocation", r"^recommendation",
    r"^crypto (investment|exchange)", r"^exchange recommendations", r"^recommended exchanges",
    r"^overall", r"^i (cannot|can't|am unable|must|appreciate|need|recommend|strongly)",
    r"^your\b", r"^- your\b", r"^here'?s why", r"^reason",
    r"^(experience|jurisdiction|regulatory|security|kyc)", r"regulatory (compliance|status|uncertainty|jurisdiction|preferences|requirements|location|verification)",
    r"kyc/aml", r"^current (market|portfolio|exchange)", r"^specific ",
    r"^(income|personal|other) ", r"^given (that|your)", r"^consider\b",
    r"^this requires", r"^as an ai", r"^market (volatility|manipulation)",
    r"^high volatility", r"^total loss", r"^due diligence", r"^critical warning",
    r"^liquidity", r"^robust ", r"^strong ", r"^institutional-grade",
    r"^research\b", r"^verify\b", r"^never invest", r"^only invest",
    r"^diversif", r"^be cautious", r"^crisis ", r"^risk-(averse|seeking|neutral)",
    r"(savings|bond|treasury|money market|index fund|real estate|insurance)",
]
NON_PRODUCT_RE = re.compile("|".join(NON_PRODUCT_PATTERNS), re.I)

REFUSAL_RE = re.compile(
    r"(\b(i cannot|i can't|i can not|cannot provide|can't provide|unable to provide|"
    r"i'm sorry|i am sorry|must (respectfully )?decline|not qualified|"
    r"cannot responsibly|i must emphasi[sz]e|consult a (licensed|qualified))\b|\bsorry,)", re.I)

CURRENCY_CODE = {
    "chf": "CHF", "sfr": "CHF", "fr": "CHF", "fr.": "CHF", "chf.": "CHF",
    "usd": "USD", "$": "USD", "us$": "USD", "usd.": "USD",
    "eur": "EUR", "\u20ac": "EUR", "eur.": "EUR",
    "gbp": "GBP", "\u00a3": "GBP",
}
PERCENT_TOKENS = {"%", "\uff05", "percent", "pct", "per cent"}
_CUR_ALT = r"chf|sfr|usd|us\$|\$|eur|\u20ac|gbp|\u00a3"
_PCT_ALT = r"%|\uff05|percent|pct"

_DIGITS = r"[\d'\u2018\u2019,\s\.]"
_DIGITS_NS = r"[\d'\u2018\u2019,\.]"
AMOUNT_RE = re.compile(rf"([\d]{_DIGITS}*)\s*({_CUR_ALT}|{_PCT_ALT})?\s*$", re.I)
RANGE_RE = re.compile(
    rf"([\d]{_DIGITS}*?)\s*(?:-|\u2013|\u2014|to)\s*([\d]{_DIGITS}*?)"
    rf"\s*({_CUR_ALT}|{_PCT_ALT})?\s*$", re.I)

LOOSE_AMOUNT_RE = re.compile(
    rf"({_CUR_ALT})?\s*(\d{_DIGITS_NS}*)\s*({_PCT_ALT}|{_CUR_ALT})?", re.I)

NON_MONETARY_TAIL_RE = re.compile(
    r"^\s*(?:/\s*\d"
    r"|x\b|btc|eth|sol|ada|xrp|dot|ltc|bnb|link|avax|sats?|satoshis?"
    r"|coins?|tokens?|shares?|units?|pieces?|assets?|positions?"
    r"|years?|yrs?|months?|mos?|weeks?|days?|quarters?|hours?)",
    re.I)

NON_MONETARY_HEAD_RE = re.compile(r"\d\s*/\s*$")

_LEAD_RE = re.compile(r"^[\*\#\-\+\>\|_\u2022\u2013\u2014\s]+")
_NUM_RE = re.compile(r"^\(?\d+[\.\)]\s*")
_TRAIL = " :-*_|>+.\u2013\u2014"
_DASH_SPLIT_RE = re.compile(r"\s+[-\u2013\u2014]\s+")
_LEAD_UNIT_RE = re.compile(r"^\s*(chf|usd|\$|eur|\u20ac)\s*", re.I)

def clean_name(s: str) -> str:
    s = _LEAD_RE.sub("", s.strip())
    s = _NUM_RE.sub("", s)
    s = s.replace("**", "").replace("`", "").strip()
    return re.sub(r"\s+", " ", s).strip(_TRAIL)

def split_line(line: str):
    if ":" in line:
        return line.split(":", 1)
    if line.startswith("|") and line.count("|") >= 2:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2:
            return cells[0], cells[1]
    m = _DASH_SPLIT_RE.search(line)
    if m:
        return line[:m.start()], line[m.end():]
    return line, ""

_SPACE_GROUPED_RE = re.compile(r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d{1,2})?")

def _to_float(raw_num: str):
    s = raw_num.strip()

    if re.search(r"[\s\u00a0\u202f]", s):
        if _SPACE_GROUPED_RE.fullmatch(s):
            s = re.sub(r"[\s\u00a0\u202f]", "", s)
        else:
            return None
    s = re.sub(r"['\u2018\u2019]", "", s)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+,\d{1,2}", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(,\d{3})+\.\d+", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    elif re.fullmatch(r"\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"\d+,\d{1,2}", s):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    if s.count(".") > 1 or s in ("", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _kind(unit: str):
    unit = (unit or "").strip().lower()
    if unit in PERCENT_TOKENS:
        return "percent"
    if unit in CURRENCY_CODE:
        return "currency"
    return "unitless"

def _currency(unit: str):
    return CURRENCY_CODE.get((unit or "").strip().lower())

_UNREADABLE = {"-", "\u2013", "\u2014", "n/a", "na", "--", "?", "none", "not specified",
               "unspecified", "tbd", "x", "$x", "chf x", "-%", "0-", "varies"}

def parse_amount(s: str, with_currency: bool = False):
    def out(v, k, note, cur):
        return (v, k, note, cur) if with_currency else (v, k, note)

    s = (s or "").strip().replace("**", "").replace("`", "")
    if not s or s.strip(" .*_").lower() in _UNREADABLE:
        return out(None, None, None, None)
    inner = re.findall(r"\((.*?)\)", s)
    s = re.sub(r"\(.*?\)", "", s).split("(")[0].strip()

    lead = _LEAD_UNIT_RE.match(s)
    lead_unit = lead.group(1) if lead else None

    m = RANGE_RE.search(s)
    if m:
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        if lo is not None and hi is not None:
            val = lo if RANGE_POLICY == "lower" else hi if RANGE_POLICY == "upper" else (lo + hi) / 2.0
            unit = m.group(3) or lead_unit
            return out(val, _kind(unit), "range", _currency(unit))

    m = AMOUNT_RE.search(s)
    if m and not NON_MONETARY_HEAD_RE.search(s[:m.start(1)]):
        val = _to_float(m.group(1))
        if val is not None:
            unit = m.group(2) or lead_unit
            return out(val, _kind(unit), None, _currency(unit))

    for cand in [s] + inner:
        has_unit_marker = bool(re.search(rf"{_CUR_ALT}|{_PCT_ALT}", cand, re.I))
        if not has_unit_marker and len(cand.split()) > LOOSE_MAX_WORDS_WITHOUT_UNIT:
            continue
        for m in LOOSE_AMOUNT_RE.finditer(cand):
            val = _to_float(m.group(2))
            if val is None:
                continue
            unit = m.group(3) or m.group(1) or lead_unit
            if not m.group(3) and (NON_MONETARY_TAIL_RE.match(cand[m.end(2):])
                                   or NON_MONETARY_HEAD_RE.search(cand[:m.start(2)])):
                continue
            return out(val, _kind(unit), "loose", _currency(unit))
    return out(None, None, None, None)

def _resolve_unitless(vals, units, budget):
    out = list(units)
    known = {u for u, v in zip(units, vals)
             if v is not None and u in ("percent", "currency")}
    todo = [i for i, u in enumerate(out) if u == "unitless"]
    if not todo:
        return out
    if known == {"percent"}:
        fill = "percent"
    elif known == {"currency"}:
        fill = "currency"
    else:
        vv = [vals[i] for i in todo if vals[i] is not None]
        tot = float(sum(vv)) if vv else 0.0
        small = bool(vv) and all(v <= UNITLESS_AS_PERCENT_MAX for v in vv)
        fill = "percent" if (small and abs(tot - 100.0) <= PCT_SUM_TOL) else "currency"
        if np.isfinite(budget) and budget > 0 and abs(tot - budget) <= 0.05 * budget:
            fill = "currency"
    for i in todo:
        out[i] = fill
    return out

def resolve_amounts(vals, units, budget):
    n = len(vals)
    if n == 0:
        e = np.zeros(0)
        return e, e, [], "none", np.nan, np.nan
    units = _resolve_unitless(vals, units, budget)
    v = np.array([np.nan if x is None else float(x) for x in vals], float)
    is_pct = np.array([u == "percent" for u in units]) & np.isfinite(v)
    is_cur = np.array([u == "currency" for u in units]) & np.isfinite(v)
    pct_sum = float(v[is_pct].sum()) if is_pct.any() else np.nan
    chf_sum = float(v[is_cur].sum()) if is_cur.any() else np.nan

    if not (is_pct.any() or is_cur.any()):
        return np.full(n, np.nan), np.full(n, np.nan), units, "none", pct_sum, chf_sum

    if np.isfinite(budget) and budget > 0:
        total, basis = float(budget), "prompt"
    elif is_pct.any() and is_cur.any():
        rest = 1.0 - min(pct_sum, 95.0) / 100.0
        total, basis = (chf_sum / rest if rest > 0 else chf_sum), "implied"
    elif is_pct.any():
        total, basis = np.nan, "proportional"
    else:
        total, basis = np.nan, "reported"

    money = np.where(is_pct, v / 100.0 * total, np.where(is_cur, v, np.nan))
    weight = np.where(np.isfinite(money), money,
                      np.where(is_pct, v, np.nan))
    return money, weight, units, basis, pct_sum, chf_sum

def rank_weights(n: int) -> np.ndarray:
    idx = np.arange(1, n + 1)
    if RANK_WEIGHTING == "uniform":
        w = np.ones(n)
    elif RANK_WEIGHTING == "zipf":
        w = 1.0 / idx
    else:
        w = (n - idx + 1).astype(float)
    return w / w.sum()

try:
    from pygini import gini as pygini_gini
    HAVE_PYGINI = True
except ImportError:
    print("\n!! pygini not installed - run:  pip install pygini")
    HAVE_PYGINI = False

def gini_paper(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return np.nan
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * x) / (n * total))

def gini_pygini(x) -> float:
    if not HAVE_PYGINI:
        return np.nan
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(pygini_gini(np.ascontiguousarray(x))) if x.size else np.nan

_rng = np.random.default_rng(0)
_maxdiff = 0.0
for _ in range(500):
    v = _rng.random(int(_rng.integers(2, 80))) * int(_rng.integers(1, 1000))
    if HAVE_PYGINI:
        _maxdiff = max(_maxdiff, abs(gini_paper(v) - gini_pygini(v)))
assert (not HAVE_PYGINI) or _maxdiff < 1e-6, _maxdiff   # silent guard

def hhi(shares) -> float:
    s = np.asarray(shares, float)
    s = s[~np.isnan(s)]
    return float(np.sum(s ** 2)) if s.size else np.nan

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 130, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "legend.frameon": False,
})

CAT_COLORS = {

    "Store of Value (PoW)": "orange", "Smart-Contract L1": "blue",
    "Layer 2 & Scaling": "cyan", "DeFi": "green",
    "AI & Infrastructure": "purple", "Payments": "pink",
    "Exchange & Platform": "gold", "RWA & Tokenized": "olive",
    "Stablecoin": "gray", "Meme": "red",

    "US-listed CEX": "blue", "EU-regulated CEX": "cyan",
    "Swiss bank/broker": "green", "Global CEX": "orange",
    "Derivatives venue": "purple", "CeFi lending": "olive",
    "Non-custodial/brokerage": "teal", "On-chain / DEX": "pink",
    "Defunct": "red",
}
PALETTE = ["blue", "red", "green", "orange", "purple", "cyan",
           "pink", "brown", "olive", "teal", "magenta", "navy"]
OTHERS_COLOR = "silver"
LABEL_COLOR = "black"

TIER_COLORS = {
    "Stable": "gray", "Blue chip": "blue", "Large-cap alt": "cyan",
    "Small-cap alt": "green", "Meme": "red",
    "Swiss regulated": "green", "EU regulated": "cyan", "US regulated": "blue",
    "Global / offshore": "orange", "Non-custodial": "purple",
    "Defunct / failed": "red",
}

SEQ_BLUE = mcolors.LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_BLUE = SEQ_BLUE.copy()
SEQ_BLUE.set_bad("#f0efec")

def _text_on(color):
    r, g, b = mcolors.to_rgb(color)
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"

def _short(model):
    return model.replace(" Flash", "").replace("Claude ", "")

def run_scenario(scen: str) -> dict:
    cfg = SCENARIO_CFG[scen]
    registry = cfg["registry"]
    CATEGORY = cfg["category"]
    CATEGORY_ORDER = [c for c in cfg["category_order"]]
    TIER_BY_CAT = cfg["tier_by_category"]
    TIER_ORDER = cfg["tier_order"]
    PROD = cfg["product_word"]

    out_dir = OUT_ROOT / scen
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save(df, name, index=False):
        df.to_csv(out_dir / name, index=index, encoding="utf-8-sig")

    def savefig(fig, name):
        fig.savefig(fig_dir / f"{name}.png", dpi=220)
        if SAVE_PDF:
            fig.savefig(fig_dir / f"{name}.pdf")
        plt.close(fig)
        print(f"  figures/{name}.png")

    alias = {}
    for canon, (code, aliases) in registry.items():
        for k in [canon.lower(), code.lower(), *[a.lower() for a in aliases]]:
            if k in alias and alias[k] != canon:
                raise KeyError(f"[{scen}] alias collision: {k!r} claimed by "
                               f"{alias[k]!r} and {canon!r}")
            alias[k] = canon
    CODE = {c: code for c, (code, _) in registry.items()}

    print("\n" + "=" * 74)
    print(f"SCENARIO: {scen.upper()}   "
          f"({len(registry)} canonical {PROD}s, {len(alias)} surface forms)")
    print("=" * 74)

    raw = db[db["scenario"] == scen].copy().reset_index(drop=True)
    print(raw.groupby("model").size().reindex(MODEL_ORDER).rename("responses").to_string())

    n_models = raw["model"].nunique()
    key_cov = raw.groupby("prompt_key")["model"].nunique()
    common_keys = set(key_cov[key_cov == n_models].index)
    print(f"prompt keys: {len(key_cov)} total, {len(common_keys)} answered by all {n_models} models")

    n_all = len(raw)
    dropped = raw[~raw["prompt_key"].isin(common_keys)]
    if len(dropped):
        save(dropped.groupby(["model", "condition"]).size()
             .rename("n_dropped").reset_index(), "paired_mode_dropped.csv")
    raw = raw[raw["prompt_key"].isin(common_keys)].reset_index(drop=True)
    raw["response_id"] = np.arange(len(raw))
    print(f"balanced panel: {n_all} -> {len(raw)} responses "
          f"({len(common_keys)} prompts x {n_models} models)")

    def parse_frame(frame: pd.DataFrame):
        records, status_rows = [], []
        unmapped, unmapped_example = Counter(), {}
        unparsed_amounts = Counter()

        for row in frame.itertuples(index=False):
            text = str(row.response)
            is_refusal = bool(REFUSAL_RE.search(text))

            items = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                name_part, amt_part = split_line(line)
                name = clean_name(name_part)
                if not name or len(name) > 45 or NON_PRODUCT_RE.search(name):
                    continue

                key = name.lower().strip(" .")
                canon = alias.get(key)
                if canon is None:
                    canon = alias.get(re.sub(r"\s*\(.*?\)\s*", " ", key).strip())
                if canon is None:
                    if re.search(r"[a-z]", key) and len(key.split()) <= 4:
                        unmapped[key] += 1
                        unmapped_example.setdefault(key, name)
                    continue

                amount, kind, note, cur = parse_amount(amt_part, with_currency=True)
                if amount is None and amt_part.strip():
                    unparsed_amounts[amt_part.strip()[:60]] += 1
                items.append((canon, name, amount, kind, note, len(items) + 1, cur))

            if items and not is_refusal:
                stat = "valid"
            elif items and is_refusal:
                stat = "hedged"
            elif not items and is_refusal:
                stat = "refusal"
            else:
                stat = "unparseable"

            budget = (float(row.budget_chf)
                      if row.budget_chf is not None and pd.notna(row.budget_chf)
                      else np.nan)
            raw_vals = [i[2] for i in items]
            raw_units = [i[3] for i in items]
            money, weight, units, unit_basis, pct_sum, chf_sum = resolve_amounts(
                raw_vals, raw_units, budget)

            kinds_seen = {u for u, v in zip(units, raw_vals)
                          if v is not None and u in ("percent", "currency")}
            alloc_chf = float(np.nansum(money)) if np.isfinite(money).any() else np.nan

            curs = {c for c, v in zip([i[6] for i in items], raw_vals)
                    if v is not None and c}
            written_cur = (sorted(curs)[0] if len(curs) == 1
                           else ("mixed" if curs else None))
            non_chf = bool(curs - {"CHF"})

            flat_budget = False
            if (FLAT_BUDGET_AS_ALTERNATIVES and len(items) > 1
                    and np.isfinite(budget) and budget > 0
                    and np.isfinite(money).all()
                    and np.allclose(money, budget, rtol=1e-6, atol=1e-9)):
                flat_budget = True

            usable_util = (np.isfinite(budget) and budget > 0
                           and np.isfinite(alloc_chf)
                           and not non_chf and not flat_budget)
            status_rows.append({
                "response_id": row.response_id, "response_uid": row.id,
                "model": row.model,
                "condition": row.condition, "status": stat,
                "n_products": len(items), "n_chars": len(text),
                "n_unparsed_amounts": sum(1 for v in raw_vals if v is None),
                "n_ranges": sum(1 for i in items if i[4] == "range"),
                "n_loose_amounts": sum(1 for i in items if i[4] == "loose"),
                "mixed_units": len(kinds_seen) > 1,
                "amount_kind": (sorted(kinds_seen)[0] if len(kinds_seen) == 1
                                else ("mixed" if kinds_seen else "none")),
                "unit_basis": unit_basis,
                "pct_sum": pct_sum,
                "budget_chf": budget,
                "amount_sum_chf": alloc_chf,
                "written_currency": written_cur,
                "non_chf_currency": non_chf,
                "flat_full_budget": flat_budget,
                "budget_utilisation": (alloc_chf / budget if usable_util else np.nan),
            })

            if not items:
                continue

            ok = np.isfinite(weight)
            w = np.where(ok, weight, 0.0)
            if flat_budget:

                shares, basis = rank_weights(len(items)), "rank_flat_budget"
            elif ok.any() and w.sum() > 0:
                shares = w / w.sum()
                basis = "reported" if ok.all() else "reported_partial"
            elif ok.any():
                shares, basis = rank_weights(len(items)), "rank_all_zero"
            else:
                shares, basis = rank_weights(len(items)), "rank_imputed"

            for idx, (canon, rawname, amt, kind, note, rank, cur) in enumerate(items):
                if (ZERO_AMOUNT_AS_REJECTION and basis.startswith("reported")
                        and ok[idx] and weight[idx] == 0):
                    continue
                records.append({
                    "response_id": row.response_id, "model": row.model, "rank": rank,
                    "product_raw": rawname, "product": canon,
                    "code": CODE.get(canon, canon),

                    "amount_reported": (float(amt) if amt is not None else np.nan),
                    "amount_unit": (units[idx] if amt is not None else "unparsed"),
                    "amount_currency": cur,

                    "amount_chf": money[idx],
                    "unit_basis": unit_basis,
                    "amount_note": note, "share": shares[idx], "alloc_basis": basis,
                })

        p = pd.DataFrame(records)
        s = pd.DataFrame(status_rows)
        if p.empty:
            raise RuntimeError(f"[{scen}] nothing parsed")

        n_before = len(p)
        p = (p.sort_values(["response_id", "rank"])
             .groupby(["response_id", "product"], as_index=False)
             .agg(model=("model", "first"), rank=("rank", "min"),
                  product_raw=("product_raw", "first"), code=("code", "first"),
                  amount_reported=("amount_reported", lambda v: v.sum(min_count=1)),
                  amount_unit=("amount_unit", "first"),
                  amount_currency=("amount_currency", "first"),
                  amount_chf=("amount_chf", lambda v: v.sum(min_count=1)),
                  unit_basis=("unit_basis", "first"),
                  amount_note=("amount_note", "first"),
                  share=("share", "sum"), alloc_basis=("alloc_basis", "first"),
                  n_lines=("rank", "size")))
        if n_before != len(p):
            print(f"  collapsed {n_before - len(p)} duplicate lines "
                  f"(same canonical {PROD} named twice in one response)")

        p["share"] = p.groupby("response_id")["share"].transform(lambda v: v / v.sum())
        meta = (["response_id", "id", "condition", "prompt_key", "attrs", "budget_chf",
                 "risk_value", "term_value", "env_value"]
                + [f"has_{a}" for a in ATTRIBUTES])
        p = p.merge(frame[meta], on="response_id", how="left")

        p = p.rename(columns={"id": "response_uid"})
        p["category"] = p["product"].map(CATEGORY)
        p["tier"] = p["category"].map(TIER_BY_CAT)
        if scen == "tokens":
            p["tier"] = np.where(p["product"].isin(BLUE_CHIPS), "Blue chip", p["tier"])
        return p, s, unmapped, unmapped_example, unparsed_amounts

    parsed, status, unmapped, unmapped_example, unparsed_amounts = parse_frame(raw)

    save(status, "response_status.csv")
    save(parsed, "parsed_recommendations.csv")

    print(f"\nparsed rows (response x {PROD}): {len(parsed):,}")
    print(f"responses with >=1 {PROD}: {parsed['response_id'].nunique():,} / {len(raw):,} "
          f"(status=='valid': {(status['status'] == 'valid').sum():,})")
    print("\nresponse status by model:")
    print(pd.crosstab(status["model"], status["status"]).reindex(MODEL_ORDER).to_string())
    print("\nallocation basis:")
    print(parsed["alloc_basis"].value_counts().to_string())
    print("\namount unit by model (how the model expressed the allocation):")
    print(pd.crosstab(status["model"], status["amount_kind"]).reindex(MODEL_ORDER).to_string())
    print(f"responses with mixed %/currency units: {int(status['mixed_units'].sum())}")
    print(f"amounts given as a range (-> {RANGE_POLICY}): {int(status['n_ranges'].sum())}")
    print(f"amounts read with the loose fallback pattern: {int(status['n_loose_amounts'].sum())}")

    cur_counts = status["written_currency"].value_counts(dropna=True)
    if len(cur_counts):
        print("\ncurrency written by the model (prompts are all in CHF):")
        print(cur_counts.to_string())
    n_nonchf = int(status["non_chf_currency"].sum())
    if n_nonchf:
        with_budget = int((status["non_chf_currency"]
                           & status["budget_chf"].notna()).sum())
        print(f"responses answering in a non-CHF currency: {n_nonchf} "
              f"({with_budget} of them against a CHF budget in the prompt)")
        print("  -> figures kept at face value (no FX rate); shares are "
              "unaffected, budget utilisation is not computed for them")
        if with_budget:
            print("  !! WARNING: a non-CHF answer to a CHF budget - check "
                  "these before quoting any CHF figure")

    n_flat = int(status["flat_full_budget"].sum())
    if n_flat:
        print(f"\nresponses repeating the FULL budget on every {PROD} "
              f"({n_flat}): read as a ranked list of alternatives, not an "
              f"allocation" if FLAT_BUDGET_AS_ALTERNATIVES else
              f"\nresponses repeating the full budget on every {PROD}: {n_flat} "
              f"(scored as written - FLAT_BUDGET_AS_ALTERNATIVES is off)")
        print(status[status["flat_full_budget"]]
              .groupby("model").size().rename("n").to_string())

    print("\nunit basis (how a percentage was turned into money):")
    print(status["unit_basis"].value_counts().to_string())
    pct_resp = status[status["pct_sum"].notna()]
    if len(pct_resp):
        print(f"responses answering in %: {len(pct_resp)}  "
              f"(sum of the percentages: median {pct_resp['pct_sum'].median():.1f}, "
              f"within 95-105: {int(pct_resp['pct_sum'].between(95, 105).sum())})")

    if unmapped:
        save(pd.DataFrame([{"surface_form": k, "example_as_written": unmapped_example[k], "n": n}
                           for k, n in unmapped.most_common()]), "unmapped_names.csv")
        print(f"{len(unmapped)} unmapped surface forms -> unmapped_names.csv")
    if unparsed_amounts:
        save(pd.DataFrame(unparsed_amounts.most_common(), columns=["amount_text", "n"]),
             "unparsed_amounts.csv")

    UNION = sorted(parsed["product"].unique())
    print(f"\ngini support = {GINI_SUPPORT}: union = {len(UNION)} {PROD}s "
          f"recommended by at least one model")

    def _vec(series, support):
        return series.reindex(support, fill_value=0).to_numpy(float)

    def gini_on(series, support):
        return gini_paper(_vec(series, support))

    def gini_rows(df):
        rows = []
        for model, sub in df.groupby("model"):
            own = sorted(sub["product"].unique())
            amt = sub.groupby("product")["share"].sum()
            frq = sub["product"].value_counts()
            a_uni, f_uni = _vec(amt, UNION), _vec(frq, UNION)
            a_own, f_own = _vec(amt, own), _vec(frq, own)
            prim_a, prim_f = ((a_uni, f_uni) if GINI_SUPPORT == "union"
                              else (a_own, f_own))
            tot_a, tot_f = float(amt.sum()), float(frq.sum())
            rows.append({
                "model": model,
                "n_responses": sub["response_id"].nunique(),
                "n_products": len(own),
                "n_products_union": len(UNION),
                "gini_support": GINI_SUPPORT,

                "GI_amount_paper": gini_paper(prim_a),
                "GI_amount_pygini": gini_pygini(prim_a),
                "GI_freq_paper": gini_paper(prim_f),
                "GI_freq_pygini": gini_pygini(prim_f),

                "GI_amount_union": gini_paper(a_uni),
                "GI_freq_union": gini_paper(f_uni),

                "GI_amount_own": gini_paper(a_own),
                "GI_freq_own": gini_paper(f_own),
                "top1_amount_share": float(amt.max() / tot_a) if tot_a > 0 else np.nan,
                "top3_amount_share": float(amt.nlargest(3).sum() / tot_a) if tot_a > 0 else np.nan,
                "top1_freq_share": float(frq.max() / tot_f) if tot_f > 0 else np.nan,
                "top3_freq_share": float(frq.nlargest(3).sum() / tot_f) if tot_f > 0 else np.nan,
                "HHI_amount": hhi(amt / tot_a) if tot_a > 0 else np.nan,
                "effective_n_amount": (1.0 / hhi(amt / tot_a)) if tot_a > 0 else np.nan,
                "mean_products_per_response": len(sub) / sub["response_id"].nunique(),
            })
        return pd.DataFrame(rows).set_index("model").reindex(MODEL_ORDER).reset_index()

    gini_by_model = gini_rows(parsed)
    save(gini_by_model, "gini_by_model.csv")
    print(f"\n=== Gini by model (balanced panel, {GINI_SUPPORT} support) ===")
    show = ["model", "n_products", "n_products_union",
            "GI_amount_paper", "GI_amount_pygini", "GI_freq_paper", "GI_freq_pygini",
            "GI_amount_own", "GI_freq_own"]
    print(gini_by_model[show].round(4).to_string(index=False))

    ver = gini_by_model[["model"]].copy()
    ver["abs_diff_amount"] = (gini_by_model["GI_amount_paper"]
                              - gini_by_model["GI_amount_pygini"]).abs()
    ver["abs_diff_freq"] = (gini_by_model["GI_freq_paper"]
                            - gini_by_model["GI_freq_pygini"]).abs()
    ver["agree_1e-9"] = (ver[["abs_diff_amount", "abs_diff_freq"]].max(axis=1) < 1e-9)
    save(ver, "gini_verification.csv")   # evidence on file, not on screen

    save(gini_by_model[["model", "n_responses", "n_products", "n_products_union",
                        "top1_amount_share", "top3_amount_share", "top1_freq_share",
                        "top3_freq_share", "HHI_amount", "effective_n_amount",
                        "mean_products_per_response"]], "concentration_summary.csv")

    st = status.merge(raw[["response_id", "prompt_key"]], on="response_id", how="left")
    st["usable"] = st["status"].isin(["valid", "hedged"])
    cov_u = (st.pivot_table(index="prompt_key", columns="model", values="usable",
                            aggfunc="max")
             .reindex(columns=MODEL_ORDER).fillna(False).astype(bool))
    paired_keys = set(cov_u.index[cov_u.all(axis=1)])

    print("\n=== robustness: differential refusal ===")
    print(f"prompts answered by every model: {len(paired_keys)} of {len(cov_u)}")
    if len(paired_keys) >= 30 and len(paired_keys) < len(cov_u):
        gini_paired = gini_rows(parsed[parsed["prompt_key"].isin(paired_keys)])
        cmp_ = (gini_by_model[["model", "GI_amount_paper", "GI_freq_paper", "n_products"]]
                .merge(gini_paired[["model", "GI_amount_paper", "GI_freq_paper",
                                    "n_products", "n_responses"]],
                       on="model", suffixes=("", "_paired")))
        cmp_ = cmp_.set_index("model").reindex(MODEL_ORDER).reset_index()
        save(cmp_, "gini_paired_usable.csv")
        print(cmp_[["model", "GI_amount_paper", "GI_amount_paper_paired",
                    "GI_freq_paper", "GI_freq_paper_paired"]]
              .round(4).to_string(index=False))
        same_rank = (list(cmp_.sort_values("GI_amount_paper", ascending=False)["model"])
                     == list(cmp_.sort_values("GI_amount_paper_paired",
                                              ascending=False)["model"]))
        if not same_rank:
            print("  !! the model ranking DEPENDS on who refused - report the "
                  "paired-usable numbers, not the headline")
    else:
        print("  every model answered (almost) every prompt - no confound to check")

    resp_with = (parsed.groupby("model")["response_id"].nunique()
                 .reindex(MODEL_ORDER, fill_value=0))
    resp_total = raw.groupby("model").size().reindex(MODEL_ORDER, fill_value=0)

    freq = (parsed.groupby(["model", "product", "code", "category", "tier"])
            .agg(n_mentions=("response_id", "size"),
                 n_responses=("response_id", "nunique"),
                 amount_share=("share", "sum"))
            .reset_index())
    freq["freq_share"] = freq["n_mentions"] / freq.groupby("model")["n_mentions"].transform("sum")
    freq["amount_share"] = freq["amount_share"] / freq.groupby("model")["amount_share"].transform("sum")
    freq = freq.sort_values(["model", "freq_share"], ascending=[True, False])
    save(freq, "recommendation_frequency.csv")

    print(f"\n=== Top-10 {PROD}s by recommendation frequency ===")
    for model in MODEL_ORDER:
        sub = freq[freq["model"] == model]
        print(f"\n--- {model} ({resp_with[model]} responses with {PROD}s "
              f"/ {resp_total[model]} total) ---")
        print(sub.head(10)[["product", "code", "category", "n_mentions", "freq_share",
                            "amount_share"]]
              .to_string(index=False, formatters={
                  "freq_share": "{:.2%}".format,
                  "amount_share": "{:.2%}".format}))

    def share_table(df, group, value, order):
        if value == "amount":
            m = df.pivot_table(index="model", columns=group, values="share", aggfunc="sum")
        else:
            m = df.pivot_table(index="model", columns=group, values="response_id", aggfunc="size")
        m = m.fillna(0.0)
        m = m.div(m.sum(axis=1), axis=0)
        cols = [c for c in order if c in m.columns]
        return m.reindex(MODEL_ORDER)[cols]

    cat_amount = share_table(parsed, "category", "amount", CATEGORY_ORDER)
    cat_freq = share_table(parsed, "category", "freq", CATEGORY_ORDER)[cat_amount.columns]
    tier_amount = share_table(parsed, "tier", "amount", TIER_ORDER)
    tier_freq = share_table(parsed, "tier", "freq", TIER_ORDER)[tier_amount.columns]

    save(cat_amount.round(4), "category_share_amount.csv", index=True)
    save(cat_freq.round(4), "category_share_frequency.csv", index=True)
    save(tier_amount.round(4), "risk_tier_share_amount.csv", index=True)
    save(tier_freq.round(4), "risk_tier_share_frequency.csv", index=True)

    print(f"\n=== {cfg['cat_word'].title()} share of investment amount (%) ===")
    print((cat_amount * 100).round(1).to_string())
    print(f"\n=== {cfg['tier_word'].title()} share of investment amount (%) ===")
    print((tier_amount * 100).round(1).to_string())

    CAT_SUPPORT = [c for c in CATEGORY_ORDER if c in set(parsed["category"])]
    rows = []
    for model, sub in parsed.groupby("model"):
        amt = sub.groupby("category")["share"].sum()
        frq = sub["category"].value_counts()
        rows.append({"model": model,
                     "GI_category_amount": gini_on(amt, CAT_SUPPORT),
                     "GI_category_freq": gini_on(frq, CAT_SUPPORT),
                     "n_categories_used": int((amt > 0).sum()),
                     "n_categories_support": len(CAT_SUPPORT)})
    gini_category = (pd.DataFrame(rows).set_index("model").reindex(MODEL_ORDER).reset_index()
                     .merge(gini_by_model[["model", "GI_amount_paper", "GI_freq_paper"]], on="model"))
    save(gini_category, "gini_by_category_level.csv")
    print(f"\n=== Gini: {PROD} level vs {cfg['cat_word']} level ({GINI_SUPPORT} support) ===")
    print(gini_category.round(4).to_string(index=False))

    VALUE_COLS = {"budget": ("budget_chf", BUDGET_ORDER),
                  "risk": ("risk_value", RISK_ORDER),
                  "term": ("term_value", TERM_ORDER),
                  "environment": ("env_value", ENV_ORDER)}
    rows = []
    for model, sub in parsed.groupby("model"):
        for a, (col, order) in VALUE_COLS.items():
            for val in order:
                s = sub[sub[col] == val]
                if s.empty:
                    continue
                amt = s.groupby("product")["share"].sum()
                frq = s["product"].value_counts()
                top = amt.idxmax() if float(amt.sum()) > 0 else None
                rows.append({"model": model, "attribute": a, "value": str(val),
                             "n_responses": s["response_id"].nunique(),
                             "n_products": int(amt.size),
                             "GI_amount": gini_on(amt, UNION),
                             "GI_freq": gini_on(frq, UNION),
                             "GI_amount_own": gini_paper(amt.to_numpy(float)),
                             f"top_{PROD}": top,
                             "top_share": float(amt.max() / amt.sum()) if float(amt.sum()) > 0 else np.nan})
    gini_val = pd.DataFrame(rows)
    save(gini_val, "gini_by_attribute_value.csv")
    print(f"\n=== GI(amount) by attribute VALUE ({GINI_SUPPORT} support) ===")
    print(gini_val.groupby(["attribute", "value"])["GI_amount"].mean()
          .round(3).to_string())

    print("\nwriting figures...")

    sets = {m: set(freq[freq["model"] == m]["product"]) for m in MODEL_ORDER}
    rows = []
    for a, b in combinations(MODEL_ORDER, 2):
        inter = len(sets[a] & sets[b])
        smaller = min(len(sets[a]), len(sets[b]))
        rows.append({"model_a": a, "model_b": b,
                     "n_a": len(sets[a]), "n_b": len(sets[b]),
                     "n_shared": inter, "n_smaller": smaller,
                     "overlap_coverage": inter / smaller if smaller else np.nan})
    overlap = pd.DataFrame(rows)
    save(overlap, "overlap_recommendations.csv")
    print(f"\n=== Pairwise {PROD} recommendation overlap between models ===")
    print("overlap_coverage = n_shared / the smaller of the two supports")
    print(overlap.round(3).to_string(index=False))

    def top_k_colors(metrics, k=3):
        seen = []
        for metric in metrics:
            for model in MODEL_ORDER:
                for t in freq[freq["model"] == model].nlargest(k, metric)["code"]:
                    if t not in seen:
                        seen.append(t)
        return {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(seen)}

    def plot_top_k(ax, metric, colors, k, title):
        x = np.arange(len(MODEL_ORDER))
        for xi, model in enumerate(MODEL_ORDER):
            sub = freq[freq["model"] == model].nlargest(k, metric)
            bottom = 0.0
            for _, r in sub.iloc[::-1].iterrows():
                c = colors[r["code"]]
                ax.bar(xi, r[metric], bottom=bottom, color=c, width=0.62,
                       edgecolor="white", linewidth=0.7)
                if r[metric] > 0.03:
                    ax.text(xi, bottom + r[metric] / 2, f"{r[metric]:.1%}",
                            ha="center", va="center", fontsize=7.5,
                            color=_text_on(c), fontweight="bold")
                bottom += r[metric]
            rest = max(0.0, 1 - bottom)
            ax.bar(xi, rest, bottom=bottom, color=OTHERS_COLOR, width=0.62,
                   edgecolor="white", linewidth=0.7)
            if rest > 0.03:
                ax.text(xi, bottom + rest / 2, f"{rest:.1%}", ha="center",
                        va="center", fontsize=7.5, color=LABEL_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)

    C3 = top_k_colors(["amount_share", "freq_share"], 3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plot_top_k(axes[0], "amount_share", C3, 3, "Top-3 Investment Amount")
    plot_top_k(axes[1], "freq_share", C3, 3, "Top-3 Recommendation Frequency")
    axes[0].set_ylabel("Proportion of amount")
    axes[1].set_ylabel("Proportion of frequency")
    handles = [Patch(facecolor=c, label=t) for t, c in C3.items()]
    handles.append(Patch(facecolor=OTHERS_COLOR, label="Others"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(f"Distribution of preferred {PROD}s in crypto investment recommendations",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, f"fig1_top3_{PROD}s")

    def plot_stack(ax, mat, colors, title, ylabel):
        bottoms = np.zeros(len(mat))
        x = np.arange(len(mat))
        for col in mat.columns:
            v = mat[col].to_numpy()
            ax.bar(x, v, bottom=bottoms, width=0.62, color=colors[col],
                   edgecolor="white", linewidth=0.7, label=col)
            for xi, (b, h) in enumerate(zip(bottoms, v)):
                if h > 0.045:
                    ax.text(xi, b + h / 2, f"{h:.0%}", ha="center", va="center",
                            fontsize=7, color=_text_on(colors[col]), fontweight="bold")
            bottoms += v
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in mat.index], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    plot_stack(axes[0], cat_amount, CAT_COLORS,
               f"{cfg['cat_word'].title()} share of investment amount", "Proportion of amount")
    plot_stack(axes[1], cat_freq, CAT_COLORS,
               f"{cfg['cat_word'].title()} share of recommendation frequency", "Proportion of frequency")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
               title=cfg["cat_word"].title(), title_fontsize=8.5)
    fig.suptitle(f"Distribution of preferred {PROD} {cfg['cat_word']}s",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig2_category_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    plot_stack(axes[0], tier_amount, TIER_COLORS,
               f"{cfg['tier_word'].title()} by investment amount", "Proportion of amount")
    plot_stack(axes[1], tier_freq, TIER_COLORS,
               f"{cfg['tier_word'].title()} by recommendation frequency", "Proportion of frequency")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
               title=cfg["tier_word"].title(), title_fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "fig3_risk_tier")

    n = len(MODEL_ORDER)
    nrow = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, model in zip(axes, MODEL_ORDER):
        sub = freq[freq["model"] == model].nlargest(10, "amount_share").iloc[::-1]
        y = np.arange(len(sub))
        ax.barh(y, sub["amount_share"], color=[CAT_COLORS[c] for c in sub["category"]],
                height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["code"], fontsize=8)
        for yi, v in zip(y, sub["amount_share"]):
            ax.text(v + 0.004, yi, f"{v:.1%}", va="center", fontsize=7.5, color=LABEL_COLOR)
        ax.set_xlim(0, max(0.05, sub["amount_share"].max() * 1.22))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(model, fontsize=9.5)
        ax.grid(axis="y", visible=False)
    for ax in axes[n:]:
        ax.axis("off")
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in CATEGORY_ORDER
               if c in set(freq["category"])]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
               title=cfg["cat_word"].title(), title_fontsize=8.5)
    fig.suptitle(f"Top-10 {PROD}s by investment amount, coloured by {cfg['cat_word']}",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig4_top10_by_category")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for model in MODEL_ORDER:
        amt = parsed[parsed["model"] == model].groupby("product")["share"].sum()
        sup = UNION if GINI_SUPPORT == "union" else sorted(amt.index)
        v = np.sort(_vec(amt, sup))
        if v.sum() <= 0:
            continue
        cum = np.concatenate([[0], np.cumsum(v) / v.sum()])
        xs = np.linspace(0, 1, len(cum))
        ax.plot(xs, cum, label=f"{_short(model)}  GI={gini_paper(v):.3f}",
                color=dict(zip(MODEL_ORDER, PALETTE))[model], lw=1.8)
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="perfect equality")
    ax.set_xlabel(f"Cumulative share of {PROD}s (least to most funded, "
                  f"{GINI_SUPPORT} support)")
    ax.set_ylabel("Cumulative share of investment amount")
    ax.set_title(f"Lorenz curves - concentration of {PROD} recommendations")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig5_lorenz")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (a, (col, order)) in zip(axes, VALUE_COLS.items()):
        sub = gini_val[gini_val["attribute"] == a]
        if sub.empty:
            ax.axis("off")
            continue
        labels = [str(v) for v in order if str(v) in set(sub["value"])]
        x = np.arange(len(labels))
        w = 0.8 / max(1, len(MODEL_ORDER))
        for j, model in enumerate(MODEL_ORDER):
            s = sub[sub["model"] == model].set_index("value").reindex(labels)
            ax.bar(x + j * w - 0.4 + w / 2, s["GI_amount"], width=w,
                   color=dict(zip(MODEL_ORDER, PALETTE))[model],
                   label=_short(model) if a == "budget" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v).replace(" ", "\n") if len(str(v)) > 12 else str(v)
                            for v in labels], fontsize=7.5, rotation=0)
        ax.set_title(a)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("GI (investment amount)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(f"Does the scenario move the bias?  GI of {PROD} recommendations "
                 "by attribute value", fontsize=11, fontweight="bold", y=1.04)
    fig.tight_layout()
    savefig(fig, "fig6_gini_by_attribute_value")

    ov = pd.DataFrame(np.nan, index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for r in overlap.itertuples(index=False):
        ov.loc[r.model_a, r.model_b] = ov.loc[r.model_b, r.model_a] = r.overlap_coverage

    k = len(MODEL_ORDER)
    labels = [_short(m) for m in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(1.15 * k + 1.5, 1.0 * k + 1.7))
    A = np.ma.masked_invalid(ov.to_numpy(float))
    ax.imshow(A, cmap=SEQ_BLUE, vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if A.mask[i, j]:
                continue
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if A[i, j] > 0.5 else "#222222")
    ax.set_xticks(range(k), labels, rotation=20, ha="right", fontsize=8.5)
    ax.set_yticks(range(k), labels, fontsize=8.5)

    ax.set_xticks(np.arange(k + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(k + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.grid(which="major", visible=False)
    ax.set_axisbelow(False)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title(f"{PROD.title()} recommendation overlap",
                 fontsize=11.5, fontweight="bold", pad=12)
    fig.tight_layout()
    savefig(fig, "fig7_pairwise_overlap")

    return dict(scen=scen, parsed=parsed, freq=freq, gini=gini_by_model,
                gini_val=gini_val, overlap=overlap, status=status,
                cat_amount=cat_amount, tier_amount=tier_amount, out_dir=out_dir)

results = {s: run_scenario(s) for s in SCENARIOS if s in set(db["scenario"])}

if len(results) > 1:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scen, r in results.items():
        g = r["gini"].copy()
        g["scenario"] = scen
        rows.append(g[["scenario", "model", "n_responses", "n_products",
                       "n_products_union", "gini_support",
                       "GI_amount_paper", "GI_amount_pygini", "GI_freq_paper",
                       "GI_freq_pygini", "GI_amount_own", "GI_freq_own",
                       "top1_amount_share", "top3_amount_share", "HHI_amount",
                       "effective_n_amount"]])
    cross = pd.concat(rows, ignore_index=True)
    cross.to_csv(OUT_ROOT / "cross_scenario_gini.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 74)
    print("CROSS-SCENARIO SUMMARY")
    print("=" * 74)
    print(cross.round(4).to_string(index=False))
    print("\nGI(amount) mean by scenario:")
    print(cross.groupby("scenario")[["GI_amount_paper", "GI_freq_paper",
                                     "n_products", "top1_amount_share"]]
          .mean().round(3).to_string())

    sub = cross.dropna(subset=["GI_amount_paper", "GI_freq_paper"])
    if len(sub) > 2:
        r = float(np.corrcoef(sub["GI_amount_paper"], sub["GI_freq_paper"])[0, 1])
        print(f"\nPearson r between GI(amount) and GI(frequency) across "
              f"model x scenario cells: {r:.3f}  (paper reports 0.85)")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    scen_list = list(results)
    x = np.arange(len(MODEL_ORDER))
    w = 0.8 / len(scen_list)
    for j, scen in enumerate(scen_list):
        s = cross[cross["scenario"] == scen].set_index("model").reindex(MODEL_ORDER)
        axes[0].bar(x + j * w - 0.4 + w / 2, s["GI_amount_paper"], width=w,
                    color=PALETTE[j], label=scen)
        axes[1].bar(x + j * w - 0.4 + w / 2, s["GI_freq_paper"], width=w,
                    color=PALETTE[j], label=scen)
        axes[2].bar(x + j * w - 0.4 + w / 2, s["n_products"], width=w,
                    color=PALETTE[j], label=scen)
    for ax, t, yl in zip(axes,
                         ["GI - investment amount", "GI - recommendation frequency",
                          "Distinct products recommended"],
                         ["Gini index", "Gini index", "count"]):
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_title(t)
        ax.set_ylabel(yl)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Tokens vs exchanges: product bias compared",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_scenario_comparison.png", dpi=220)
    if SAVE_PDF:
        fig.savefig(fig_dir / "fig7_scenario_comparison.pdf")
    plt.close(fig)
    print(f"  figures/fig7_scenario_comparison.png")

OUT_ROOT.mkdir(parents=True, exist_ok=True)
_merge_log.to_csv(OUT_ROOT / "data_sources.csv", index=False, encoding="utf-8-sig")
print(f"\n  data_sources.csv  ({len(_sources)} source db(s))")

print(f"\ndone -> {OUT_ROOT}")
