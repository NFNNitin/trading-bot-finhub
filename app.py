"""
╔══════════════════════════════════════════════════════════════╗
║          PRO AI TRADING DASHBOARD  —  powered by yfinance   ║
║  EMA 100/200 | 15m Scalping | 30m+1h Intraday | Backtesting ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy import stats
import time

try:
    import feedparser
    _FEED = True
except ImportError:
    _FEED = False

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Pro AI Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #080c12; }

/* Ticker card */
.ticker-card {
    background: linear-gradient(135deg,#0f1923 0%,#131f2e 100%);
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 6px;
    transition: border-color .2s;
}
.ticker-card:hover { border-color: #2a9d8f; }
.ticker-name  { font-size:11px; color:#6b8cad; letter-spacing:.08em; text-transform:uppercase; }
.ticker-price { font-family:'Space Mono',monospace; font-size:20px; font-weight:700; color:#e8f4f8; margin:4px 0; }
.ticker-chg   { font-size:13px; font-weight:600; }
.up   { color:#2a9d8f; }
.down { color:#e76f51; }

/* Master signal cards */
.msig {
    border-radius:16px;
    padding:24px 20px;
    text-align:center;
    box-shadow: 0 8px 32px rgba(0,0,0,.4);
    transition: transform .15s;
}
.msig:hover { transform: translateY(-2px); }
.msig-label { font-size:12px; letter-spacing:.12em; text-transform:uppercase; opacity:.8; margin-bottom:4px; }
.msig-signal { font-family:'Space Mono',monospace; font-size:26px; font-weight:700; margin:8px 0; }
.msig-score  { font-size:14px; opacity:.85; }

.sig-strong-buy  { background:linear-gradient(135deg,#1a4731,#2a9d8f); color:#c8ffe8; }
.sig-buy         { background:linear-gradient(135deg,#1a3a2a,#4caf7d); color:#d0ffe0; }
.sig-neutral     { background:linear-gradient(135deg,#2a2a1a,#a08c2a); color:#fff5c0; }
.sig-sell        { background:linear-gradient(135deg,#3a1a1a,#c0522a); color:#ffd0c0; }
.sig-strong-sell { background:linear-gradient(135deg,#3a0f0f,#e76f51); color:#ffc8c8; }

/* TF column */
.tf-box {
    background:#0f1923;
    border:1px solid #1e2d3d;
    border-radius:10px;
    padding:12px;
    margin-bottom:4px;
    font-size:13px;
}

/* Trade box */
.trade-long  { background:linear-gradient(180deg,#0d2318 0%,#0a1a12 100%); border:1px solid #2a9d8f; border-radius:12px; padding:16px; }
.trade-short { background:linear-gradient(180deg,#230d0d 0%,#1a0a0a 100%); border:1px solid #e76f51; border-radius:12px; padding:16px; }
.trade-none  { background:#0f1923; border:1px solid #1e2d3d; border-radius:12px; padding:16px; color:#6b8cad; text-align:center; }

/* Conflict badges */
.badge-crit   { background:#3a0f0f; border-left:3px solid #e76f51; border-radius:6px; padding:10px 14px; margin:6px 0; font-size:13px; }
.badge-warn   { background:#2a2200; border-left:3px solid #e9c46a; border-radius:6px; padding:10px 14px; margin:6px 0; font-size:13px; }
.badge-ok     { background:#0a2318; border-left:3px solid #2a9d8f; border-radius:6px; padding:10px 14px; margin:6px 0; font-size:13px; }

/* News card */
.news-card { background:#0f1923; border:1px solid #1e2d3d; border-radius:8px; padding:12px 14px; margin-bottom:8px; }
.news-title { font-size:13px; font-weight:600; color:#d0e8f8; margin-bottom:4px; line-height:1.4; }
.news-meta  { font-size:11px; color:#4a6a8a; }

/* Divider */
hr { border-color:#1e2d3d !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
_defaults = {
    "symbol": "BTC-USD",
    "sym1": "BTC-USD",
    "sym2": "GC=F",
    "view": "Single Asset",
    "show_bt": False,
    "refreshed_at": datetime.now(),
    "auto": True,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Symbol catalogue ──────────────────────────────────────────
QUICK = {
    "BTC":    "BTC-USD",
    "ETH":    "ETH-USD",
    "XRP":    "XRP-USD",
    "Gold":   "GC=F",
    "Silver": "SI=F",
    "Oil":    "CL=F",
    "EUR/USD":"EURUSD=X",
    "DXY":    "DX-Y.NYB",
    "S&P 500":"^GSPC",
    "NVDA":   "NVDA",
}

TICKER_ROW = {
    "Bitcoin": "BTC-USD",
    "Gold":    "GC=F",
    "Silver":  "SI=F",
    "DXY":     "DX-Y.NYB",
    "XRP":     "XRP-USD",
}

# ── Data layer ────────────────────────────────────────────────
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the MultiIndex yfinance returns for single symbols,
    keep only OHLCV, drop NaNs.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance ≥0.2.38 returns MultiIndex (Attribute, Ticker) for single symbols too
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Keep standard columns only
    wanted = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[wanted].copy()
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df.dropna()


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a clean OHLCV DataFrame using the dict agg syntax (works on all pandas versions)."""
    agg_dict = {
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }
    return df.resample(rule).agg(agg_dict).dropna()


@st.cache_data(ttl=60)
def fetch_data(symbol: str) -> dict | None:
    """Fetch 15m, 30m, 1h candles plus daily for 24h metrics."""
    try:
        raw_15m = yf.download(symbol, period="5d",  interval="15m", progress=False, auto_adjust=True)
        raw_1h  = yf.download(symbol, period="30d", interval="1h",  progress=False, auto_adjust=True)
        raw_1d  = yf.download(symbol, period="6mo", interval="1d",  progress=False, auto_adjust=True)

        raw_15m = _clean_df(raw_15m)
        raw_1h  = _clean_df(raw_1h)
        raw_1d  = _clean_df(raw_1d)

        if raw_15m.empty or raw_1h.empty:
            st.error(f"No data returned for **{symbol}**. "
                     "Check the symbol is correct (e.g. `BTC-USD`, `GC=F`, `AAPL`).")
            return None

        data = {
            "15m": raw_15m,
            "30m": _resample_ohlcv(raw_15m, "30min"),
            "1h":  raw_1h,
            "1d":  raw_1d,
        }
        return data
    except Exception as e:
        st.error(f"Download error for **{symbol}**: {e}")
        return None


@st.cache_data(ttl=60)
def fetch_ticker_row() -> dict:
    """Live quote for the five headline assets."""
    result = {}
    for name, sym in TICKER_ROW.items():
        try:
            raw = yf.download(sym, period="2d", interval="1d",
                              progress=False, auto_adjust=True)
            h = _clean_df(raw)
            if len(h) >= 2:
                price = float(h["Close"].iloc[-1])
                prev  = float(h["Close"].iloc[-2])
                chg   = (price - prev) / prev * 100
            elif len(h) == 1:
                price = float(h["Close"].iloc[-1])
                chg   = 0.0
            else:
                price, chg = 0.0, 0.0
            result[name] = {"price": price, "change": chg}
        except Exception:
            result[name] = {"price": 0.0, "change": 0.0}
    return result


# ── Indicators (EMA 100 + EMA 200 only for trend, plus RSI/MACD/ATR/ADX/BB/Volume) ──
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if n < 30:
        return df

    # ── THE ONLY TWO EMAs ──
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # MACD
    e12 = df["Close"].ewm(span=12, adjust=False).mean()
    e26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = e12 - e26
    df["MACD_Sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Sig"]

    # ATR  — use numpy to avoid any Series alignment issues
    hi  = df["High"].values
    lo  = df["Low"].values
    cl  = df["Close"].values
    hl  = hi - lo
    hcp = np.abs(hi[1:] - cl[:-1])
    lcp = np.abs(lo[1:] - cl[:-1])
    tr  = np.concatenate([[hl[0]], np.maximum(hl[1:], np.maximum(hcp, lcp))])
    atr_series = pd.Series(tr, index=df.index).rolling(14).mean()
    df["ATR"] = atr_series

    # ADX
    pdm = df["High"].diff().clip(lower=0)
    ndm = (-df["Low"].diff()).clip(lower=0)
    atr14 = df["ATR"]
    pdi = 100 * pdm.rolling(14).mean() / (atr14 + 1e-9)
    ndi = 100 * ndm.rolling(14).mean() / (atr14 + 1e-9)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    df["ADX"] = dx.rolling(14).mean()

    # Bollinger Bands
    df["BB_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_up"]  = df["BB_mid"] + 2 * bb_std
    df["BB_lo"]  = df["BB_mid"] - 2 * bb_std

    # Volume ratio
    df["Vol_MA"]  = df["Volume"].rolling(20).mean()
    df["Vol_Rat"] = df["Volume"] / (df["Vol_MA"] + 1e-9)

    # Stochastic
    lo14 = df["Low"].rolling(14).min()
    hi14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - lo14) / (hi14 - lo14 + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    return df


# ── Candle pattern ────────────────────────────────────────────
def candle_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "—"
    r, p, p2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    body  = abs(r.Open - r.Close)
    hi_w  = r.High - max(r.Open, r.Close)
    lo_w  = min(r.Open, r.Close) - r.Low
    rng   = r.High - r.Low + 1e-9
    p_body = abs(p.Open - p.Close)

    if body < rng * 0.1:                                          return "➕ Doji"
    if lo_w > body * 2 and hi_w < body * 0.5:                    return "🔨 Hammer"
    if hi_w > body * 2 and lo_w < body * 0.5:                    return "🌠 Shooting Star"
    if (r.Close > r.Open and p.Close < p.Open
            and r.Open <= p.Close and r.Close >= p.Open):         return "🟢 Bull Engulf"
    if (r.Close < r.Open and p.Close > p.Open
            and r.Open >= p.Close and r.Close <= p.Open):         return "🔴 Bear Engulf"
    if (p2.Close < p2.Open and p_body < abs(p2.Open-p2.Close)*0.3
            and r.Close > r.Open
            and r.Close > (p2.Open+p2.Close)/2):                  return "⭐ Morning Star"
    if (p2.Close > p2.Open and p_body < abs(p2.Open-p2.Close)*0.3
            and r.Close < r.Open
            and r.Close < (p2.Open+p2.Close)/2):                  return "🌙 Evening Star"
    return "• Normal"


# ── Simplified signal (EMA 100/200 centred) ──────────────────
def ema_signal(df: pd.DataFrame) -> dict | None:
    if len(df) < 205:
        return None
    c = df.iloc[-1]
    p = df.iloc[-2]
    score = 0

    # EMA 200 position (40 pts)
    if c.Close > c.EMA200:  score += 40
    else:                   score -= 40

    # EMA 100 position (30 pts)
    if c.Close > c.EMA100:  score += 30
    else:                   score -= 30

    # EMA alignment (20 pts)
    if c.EMA100 > c.EMA200: score += 20
    else:                   score -= 20

    # RSI (10 pts)
    if 40 <= c.RSI <= 60:   score += 10
    elif c.RSI > 70:        score -= 5
    elif c.RSI < 30:        score += 5

    norm = (score + 100) / 200 * 100   # 0‥100

    if   norm >= 75: sig = "🟢 STRONG BUY"
    elif norm >= 60: sig = "🟢 BUY"
    elif norm >= 45: sig = "🟡 NEUTRAL"
    elif norm >= 30: sig = "🔴 SELL"
    else:            sig = "🔴 STRONG SELL"

    return dict(
        Signal=sig, Score=round(norm, 1),
        RSI=round(c.RSI, 1), ADX=round(c.ADX, 1),
        ATR=c.ATR, Price=c.Close,
        EMA100=c.EMA100, EMA200=c.EMA200,
        MACD=c.MACD, MACD_Sig=c.MACD_Sig,
    )


# ── Conflict detection ────────────────────────────────────────
def detect_conflicts(data: dict, sigs: dict) -> dict:
    df5  = add_indicators(data["15m"])
    df1h = add_indicators(data["1h"])

    p = df5["Close"]
    mom_15m = (p.iloc[-1] - p.iloc[-2]) / p.iloc[-2] * 100 if len(p) > 1 else 0
    mom_1h  = (df1h["Close"].iloc[-1] - df1h["Close"].iloc[-3]) / df1h["Close"].iloc[-3] * 100 if len(df1h) > 3 else 0

    conflicts, warnings = [], []
    risk = 0
    sig = sigs.get("15m", {})

    # Price vs signal divergence
    if sig and "BUY" in sig.get("Signal", ""):
        if mom_15m < -0.4:
            conflicts.append({"msg": f"⚡ BUY signal but price fell {mom_15m:.2f}% in last 15 min", "sev": "CRIT"})
            risk += 30
        if mom_1h < -1.0:
            conflicts.append({"msg": f"🚨 BUY signal but –{abs(mom_1h):.2f}% on 1h momentum", "sev": "CRIT"})
            risk += 25
    if sig and "SELL" in sig.get("Signal", ""):
        if mom_15m > 0.4:
            conflicts.append({"msg": f"⚡ SELL signal but price rose +{mom_15m:.2f}% in last 15 min", "sev": "CRIT"})
            risk += 25

    # TF disagreement
    s15 = sigs.get("15m", {}).get("Signal", "")
    s1h = sigs.get("1h",  {}).get("Signal", "")
    if "BUY" in s15 and "SELL" in s1h:
        conflicts.append({"msg": "⚠️ 15m BUY vs 1h SELL — counter-trend scalp", "sev": "HIGH"})
        risk += 15

    # Overbought on BUY
    if sig and "BUY" in sig.get("Signal", "") and sig.get("RSI", 50) > 72:
        warnings.append({"msg": f"RSI overbought ({sig['RSI']:.0f}) on a BUY — late entry risk"})
        risk += 8

    # Oversold on SELL
    if sig and "SELL" in sig.get("Signal", "") and sig.get("RSI", 50) < 28:
        warnings.append({"msg": f"RSI oversold ({sig['RSI']:.0f}) on a SELL — bounce risk"})
        risk += 8

    # Weak ADX
    if sig and sig.get("ADX", 30) < 18 and ("STRONG" in sig.get("Signal", "")):
        warnings.append({"msg": f"ADX {sig.get('ADX', 0):.1f} — trend is weak, signal may be noise"})
        risk += 6

    # Low volume
    last_vol = df5["Vol_Rat"].iloc[-1] if "Vol_Rat" in df5 else 1
    if last_vol < 0.5:
        warnings.append({"msg": f"Volume only {last_vol:.1f}× average — low conviction"})
        risk += 5

    if risk >= 30:   assessment, col = "🚫 HIGH RISK — avoid", "red"
    elif risk >= 15: assessment, col = "⚠️ MEDIUM RISK — reduce size", "orange"
    elif risk > 0:   assessment, col = "💛 LOW RISK — trade with care", "yellow"
    else:            assessment, col = "✅ CLEAN — signals aligned", "green"

    return dict(conflicts=conflicts, warnings=warnings,
                risk=risk, assessment=assessment, color=col,
                mom_15m=mom_15m, mom_1h=mom_1h)


# ── Master signal ─────────────────────────────────────────────
def master_signal(data: dict) -> dict:
    df15 = add_indicators(data["15m"])
    df30 = add_indicators(data["30m"])
    df1h = add_indicators(data["1h"])

    if len(df15) < 205 or len(df1h) < 205:
        return {"scalp": None, "intra": None}

    c15, c30, c1h = df15.iloc[-1], df30.iloc[-1], df1h.iloc[-1]

    # ── SCALPING (15m only, EMA 100/200) ──────────────────────
    sc, sr = 0, []
    if c15.Close > c15.EMA200 and c15.Close > c15.EMA100:
        sc += 35; sr.append("✅ Above both EMAs (strong bull zone)")
    elif c15.Close > c15.EMA200:
        sc += 18; sr.append("⚠️ Above EMA200 but below EMA100")
    else:
        sr.append("❌ Below EMA200 (bearish zone)")
    if c15.EMA100 > c15.EMA200:
        sc += 25; sr.append("✅ EMA100 > EMA200 — bullish crossover")
    else:
        sr.append("❌ EMA100 < EMA200 — bearish alignment")
    if 42 <= c15.RSI <= 62:
        sc += 20; sr.append(f"✅ RSI neutral {c15.RSI:.0f} — room to run")
    elif c15.RSI > 72:
        sr.append(f"⚠️ RSI overbought {c15.RSI:.0f}")
    elif c15.RSI < 28:
        sc += 10; sr.append(f"💎 RSI oversold {c15.RSI:.0f}")
    if c15.Vol_Rat > 1.3:
        sc += 10; sr.append(f"✅ Volume {c15.Vol_Rat:.1f}× — conviction present")
    if c15.MACD > c15.MACD_Sig:
        sc += 10; sr.append("✅ MACD bullish crossover on 15m")
    else:
        sr.append("⚠️ MACD below signal on 15m")

    # ── INTRADAY (30m + 1h, EMA 100/200 aligned) ─────────────
    ic, ir = 0, []
    # 30m leg
    if c30.Close > c30.EMA200:
        ic += 20; ir.append("✅ 30m above EMA200")
    else:
        ir.append("❌ 30m below EMA200")
    if c30.EMA100 > c30.EMA200:
        ic += 15; ir.append("✅ 30m EMA100>EMA200 bullish stack")
    # 1h leg (heavier weight)
    if c1h.Close > c1h.EMA200 and c1h.EMA100 > c1h.EMA200:
        ic += 35; ir.append("✅ 1h perfect EMA alignment")
    elif c1h.Close > c1h.EMA200:
        ic += 20; ir.append("⚠️ 1h above EMA200 but EMA100 not crossed")
    else:
        ir.append("❌ 1h below EMA200 — bearish trend")
    if c1h.ADX > 25:
        ic += 20; ir.append(f"✅ ADX {c1h.ADX:.0f} — strong trend")
    elif c1h.ADX > 18:
        ic += 10; ir.append(f"⚠️ ADX {c1h.ADX:.0f} — moderate trend")
    else:
        ir.append(f"❌ ADX {c1h.ADX:.0f} — choppy, no clear trend")
    if c1h.MACD > c1h.MACD_Sig:
        ic += 10; ir.append("✅ 1h MACD bullish")

    def _grade(s):
        if s >= 80: return "STRONG BUY",   "sig-strong-buy",  "🚀"
        if s >= 62: return "BUY",           "sig-buy",         "📈"
        if s >= 42: return "NEUTRAL",       "sig-neutral",     "⏸️"
        if s >= 25: return "SELL",          "sig-sell",        "📉"
        return           "STRONG SELL",     "sig-strong-sell", "🔻"

    scalp_sig, scalp_cls, scalp_ico = _grade(sc)
    intra_sig, intra_cls, intra_ico = _grade(ic)

    return {
        "scalp":  dict(signal=scalp_sig, cls=scalp_cls, icon=scalp_ico, score=sc,  reasons=sr),
        "intra":  dict(signal=intra_sig, cls=intra_cls, icon=intra_ico, score=ic, reasons=ir),
    }


# ── Prediction engine ─────────────────────────────────────────
def predict(df: pd.DataFrame, tf: str) -> dict | None:
    if len(df) < 60:
        return None
    price = float(df["Close"].iloc[-1])
    atr   = float(df["ATR"].iloc[-1]) if "ATR" in df else price * 0.01

    # Weighted linear regression (recent data counts more)
    recent = df["Close"].tail(30).values
    x = np.arange(len(recent))
    w = np.exp(x / len(recent))
    wx = np.average(x, weights=w); wy = np.average(recent, weights=w)
    denom = np.sum(w * (x - wx) ** 2)
    slope = np.sum(w * (x - wx) * (recent - wy)) / (denom + 1e-9)
    intercept = wy - slope * wx
    pred_lr = slope * len(recent) + intercept

    # EMA momentum
    ema_mom = df["EMA100"].iloc[-1] - df["EMA200"].iloc[-1] if "EMA100" in df else 0
    pred_ema = price + ema_mom * 0.4

    # Mean reversion
    if "BB_lo" in df:
        if price < df["BB_lo"].iloc[-1]:   pred_bb = df["BB_mid"].iloc[-1]
        elif price > df["BB_up"].iloc[-1]: pred_bb = df["BB_mid"].iloc[-1]
        else:                              pred_bb = price
    else:
        pred_bb = price

    ensemble = np.mean([pred_lr, pred_ema, pred_bb])
    move_pct  = (ensemble - price) / price * 100

    return dict(
        current=price,
        predicted=ensemble,
        move_pct=move_pct,
        upper=price + atr * 1.5,
        lower=price - atr * 1.5,
        direction="📈 UP" if move_pct > 0 else "📉 DOWN",
        strength="Strong" if abs(move_pct) > 1 else "Moderate" if abs(move_pct) > 0.3 else "Weak",
    )


# ── Trade calculator ──────────────────────────────────────────
def trade_setup(price, atr, direction="LONG", style="Scalp", rr=1.5):
    m   = 1.5 if style == "Scalp" else 2.0
    sl  = atr * m
    if direction == "LONG":
        return dict(entry=price, sl=price-sl, tp=price+sl*rr,
                    risk_pct=sl/price*100, reward_pct=sl*rr/price*100)
    return dict(entry=price, sl=price+sl, tp=price-sl*rr,
                risk_pct=sl/price*100, reward_pct=sl*rr/price*100)


# ── Back-tester ───────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, periods_ahead=1) -> dict | None:
    df = df.copy()
    df = add_indicators(df)
    if len(df) < 210:
        return None

    n_test    = min(150, len(df) - 210)
    correct   = 0
    in_range  = 0
    errors    = []
    preds_arr = []
    acts_arr  = []

    for i in range(210, 210 + n_test):
        hist = df.iloc[:i]
        p    = predict(hist, "bt")
        if p is None:
            continue
        actual = float(df["Close"].iloc[i + periods_ahead - 1])
        pred_dir  = 1 if p["predicted"] > p["current"] else -1
        act_dir   = 1 if actual > p["current"] else -1
        if pred_dir == act_dir:
            correct += 1
        if p["lower"] <= actual <= p["upper"]:
            in_range += 1
        err = abs(actual - p["predicted"]) / actual * 100
        errors.append(err)
        preds_arr.append(p["predicted"])
        acts_arr.append(actual)

    total = len(errors)
    if total == 0:
        return None

    dir_acc   = correct  / total * 100
    range_acc = in_range / total * 100
    mape      = np.mean(errors)
    recent_n  = min(20, total)
    recent_ok = sum(
        1 for i in range(-recent_n, 0)
        if (preds_arr[i] > float(df["Close"].iloc[210 + total + i - 1]))
        == (acts_arr[i]  > float(df["Close"].iloc[210 + total + i - 1]))
    )
    recent_acc = recent_ok / recent_n * 100

    grade = ("🏆 Excellent" if dir_acc >= 70
             else "✅ Good"    if dir_acc >= 60
             else "⚠️ Fair"   if dir_acc >= 50
             else "❌ Poor")

    return dict(
        dir_acc=dir_acc, range_acc=range_acc,
        mape=mape, recent_acc=recent_acc,
        total=total, grade=grade,
        preds=preds_arr[-50:], acts=acts_arr[-50:],
    )


# ── News feed ─────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_news() -> list:
    if not _FEED:
        return [{"title": "Install feedparser for live news: pip install feedparser", "link": "#", "time": ""}]
    items = []
    for url in ["https://cointelegraph.com/rss", "https://cryptonews.com/news/feed/"]:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:4]:
                items.append({"title": e.title, "link": e.link, "time": e.get("published", "")})
        except:
            pass
    return items[:10] or [{"title": "News unavailable", "link": "#", "time": ""}]


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
st.sidebar.markdown("## ⚙️ Dashboard Settings")

view = st.sidebar.radio("View Mode", ["Single Asset", "Multi-Asset"], index=0 if st.session_state.view == "Single Asset" else 1)
st.session_state.view = view

st.sidebar.divider()

if view == "Single Asset":
    sym_in = st.sidebar.text_input("Symbol", value=st.session_state.symbol, help="e.g. BTC-USD, GC=F, AAPL, EURUSD=X")
    if sym_in.upper() != st.session_state.symbol:
        st.session_state.symbol = sym_in.upper()
        st.cache_data.clear()
        st.rerun()
else:
    s1 = st.sidebar.text_input("Symbol 1", value=st.session_state.sym1)
    s2 = st.sidebar.text_input("Symbol 2", value=st.session_state.sym2)
    if s1.upper() != st.session_state.sym1:
        st.session_state.sym1 = s1.upper(); st.cache_data.clear(); st.rerun()
    if s2.upper() != st.session_state.sym2:
        st.session_state.sym2 = s2.upper(); st.cache_data.clear(); st.rerun()

st.sidebar.markdown("**⚡ Quick Select**")
qcols = st.sidebar.columns(2)
for i, (label, ticker) in enumerate(QUICK.items()):
    with qcols[i % 2]:
        if st.button(label, key=f"q_{ticker}", use_container_width=True):
            if view == "Single Asset":
                st.session_state.symbol = ticker
            else:
                st.session_state.sym1 = ticker
            st.cache_data.clear()
            st.rerun()

st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.cache_data.clear()
    st.session_state.refreshed_at = datetime.now()
    st.rerun()

st.session_state.auto = st.sidebar.checkbox("Auto-refresh (60 s)", value=st.session_state.auto)

if st.sidebar.checkbox("📊 Show Backtest"):
    st.session_state.show_bt = True
else:
    st.session_state.show_bt = False

st.sidebar.divider()
st.sidebar.markdown("**Risk Settings**")
rr_ratio = st.sidebar.slider("Risk : Reward", 1.0, 3.0, 1.5, 0.5)
pos_size = st.sidebar.number_input("Position Size ($)", 100, 1_000_000, 1000, 100)
st.sidebar.caption(f"Refreshed: {st.session_state.refreshed_at.strftime('%H:%M:%S')}")

# ── Supported symbols helper ───────────────────────────────────
with st.sidebar.expander("📖 Symbol Guide"):
    st.markdown("""
| Asset | Symbol |
|---|---|
| Bitcoin | `BTC-USD` |
| Ethereum | `ETH-USD` |
| XRP | `XRP-USD` |
| Gold | `GC=F` |
| Silver | `SI=F` |
| Oil (WTI) | `CL=F` |
| EUR/USD | `EURUSD=X` |
| DXY | `DX-Y.NYB` |
| S&P 500 | `^GSPC` |
| Any stock | `AAPL`, `TSLA` … |
""")


# ═══════════════════════════════════════════════════════════════
#  RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def render_ticker_row():
    prices = fetch_ticker_row()
    cols   = st.columns(5)
    for i, (name, d) in enumerate(prices.items()):
        chg, price = d["change"], d["price"]
        color = "up" if chg >= 0 else "down"
        arrow = "▲" if chg >= 0 else "▼"
        pf    = f"${price:,.4f}" if price < 5 else f"${price:,.2f}"
        with cols[i]:
            st.markdown(f"""
            <div class="ticker-card">
                <div class="ticker-name">{name}</div>
                <div class="ticker-price">{pf}</div>
                <div class="ticker-chg {color}">{arrow} {abs(chg):.2f}%</div>
            </div>""", unsafe_allow_html=True)


def render_master_signals(ms: dict):
    st.subheader("🎯 Master Signals")
    st.caption("Every indicator, volume, pattern, momentum & conflict factored in — one definitive signal per style")
    c1, c2 = st.columns(2)
    for col, key, label, tf_note in [
        (c1, "scalp", "⚡ SCALPING",   "15m candles"),
        (c2, "intra", "📅 INTRADAY",   "30m + 1h candles"),
    ]:
        sig = ms.get(key)
        if sig is None:
            with col:
                st.info(f"{label} — not enough data (need 200+ candles)")
            continue
        with col:
            st.markdown(f"""
            <div class="msig {sig['cls']}">
                <div class="msig-label">{label} &nbsp;·&nbsp; {tf_note}</div>
                <div class="msig-signal">{sig['icon']} {sig['signal']}</div>
                <div class="msig-score">Score {sig['score']}/100</div>
            </div>""", unsafe_allow_html=True)
            with st.expander("📋 Why this signal?"):
                for r in sig["reasons"]:
                    st.write(r)


def render_conflict_panel(cf: dict):
    st.subheader("🔬 Signal Quality Check")
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        badge = ("badge-crit" if cf["color"] == "red"
                 else "badge-warn" if cf["color"] in ("orange","yellow")
                 else "badge-ok")
        st.markdown(f'<div class="{badge}"><b>{cf["assessment"]}</b></div>', unsafe_allow_html=True)
    with c2:
        st.metric("Risk Score", cf["risk"], delta="lower=better", delta_color="inverse")
    with c3:
        st.metric("15m Momentum", f"{cf['mom_15m']:+.2f}%")

    if cf["conflicts"]:
        for c in cf["conflicts"]:
            level = "badge-crit" if c["sev"] == "CRIT" else "badge-warn"
            st.markdown(f'<div class="{level}">{c["msg"]}</div>', unsafe_allow_html=True)
    if cf["warnings"]:
        for w in cf["warnings"]:
            st.markdown(f'<div class="badge-warn">⚠️ {w["msg"]}</div>', unsafe_allow_html=True)
    if not cf["conflicts"] and not cf["warnings"]:
        st.markdown('<div class="badge-ok">✅ No conflicts detected — clean setup</div>', unsafe_allow_html=True)

    with st.expander("📊 Momentum breakdown"):
        st.write(f"15m momentum: **{cf['mom_15m']:+.2f}%**")
        st.write(f"1h  momentum: **{cf['mom_1h']:+.2f}%**")
        st.caption("If signal says BUY but momentum is negative → indicators are lagging. WAIT.")


def render_tf_scanner(data: dict) -> dict:
    st.subheader("⏰ Multi-Timeframe Scanner")
    tfs   = ["15m", "30m", "1h"]
    cols  = st.columns(3)
    sigs  = {}
    for i, tf in enumerate(tfs):
        df  = add_indicators(data[tf])
        sig = ema_signal(df)
        pat = candle_pattern(df)
        sigs[tf] = sig
        with cols[i]:
            st.markdown(f"**{tf.upper()}**")
            st.caption(pat)
            if sig:
                colour = ("#2a9d8f" if "BUY" in sig["Signal"]
                          else "#e76f51" if "SELL" in sig["Signal"]
                          else "#a08c2a")
                st.markdown(
                    f'<span style="color:{colour};font-weight:700">{sig["Signal"]}</span>',
                    unsafe_allow_html=True)
                st.progress(sig["Score"] / 100)
                st.caption(f"Score {sig['Score']}/100  ·  RSI {sig['RSI']}  ·  ADX {sig['ADX']}")
            else:
                st.caption("Not enough data")
    return sigs


def render_trade_setups(sigs: dict, rr: float, pos: int):
    st.subheader("🎯 AI Trade Setups")
    c1, c2 = st.columns(2)

    pairs = [
        (c1, "15m", "⚡ Scalping (15m)",  "Scalp"),
        (c2, "1h",  "📅 Intraday (1h)",   "Intraday"),
    ]
    for col, tf, title, style in pairs:
        sig = sigs.get(tf)
        with col:
            st.markdown(f"**{title}**")
            if sig is None:
                st.markdown('<div class="trade-none">Not enough data</div>', unsafe_allow_html=True)
                continue
            price, atr = sig["Price"], sig["ATR"]
            if "BUY" in sig["Signal"]:
                t = trade_setup(price, atr, "LONG", style, rr)
                risk_amt = pos * t["risk_pct"] / 100
                st.markdown(f"""
                <div class="trade-long">
                    <b>📈 LONG</b><br>
                    Entry &nbsp; <b>${t['entry']:,.2f}</b><br>
                    🎯 TP &nbsp;&nbsp; <b>${t['tp']:,.2f}</b> &nbsp;(+{t['reward_pct']:.2f}%)<br>
                    🛑 SL &nbsp;&nbsp; <b>${t['sl']:,.2f}</b> &nbsp;(-{t['risk_pct']:.2f}%)<br>
                    💰 Risk &nbsp; <b>${risk_amt:.2f}</b> → Reward <b>${risk_amt*rr:.2f}</b>
                </div>""", unsafe_allow_html=True)
            elif "SELL" in sig["Signal"]:
                t = trade_setup(price, atr, "SHORT", style, rr)
                risk_amt = pos * t["risk_pct"] / 100
                st.markdown(f"""
                <div class="trade-short">
                    <b>📉 SHORT</b><br>
                    Entry &nbsp; <b>${t['entry']:,.2f}</b><br>
                    🎯 TP &nbsp;&nbsp; <b>${t['tp']:,.2f}</b> &nbsp;(+{t['reward_pct']:.2f}%)<br>
                    🛑 SL &nbsp;&nbsp; <b>${t['sl']:,.2f}</b> &nbsp;(-{t['risk_pct']:.2f}%)<br>
                    💰 Risk &nbsp; <b>${risk_amt:.2f}</b> → Reward <b>${risk_amt*rr:.2f}</b>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="trade-none">⏸️ No setup — wait for clear signal</div>',
                            unsafe_allow_html=True)


def render_chart(data: dict, symbol: str):
    st.subheader("📈 Price Chart — EMA 100 & EMA 200 (1h)")
    df = add_indicators(data["1h"])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df.Open, high=df.High,
        low=df.Low, close=df.Close, name="Price",
        increasing_line_color="#2a9d8f", decreasing_line_color="#e76f51",
    ), row=1, col=1)

    # EMA 100
    fig.add_trace(go.Scatter(
        x=df.index, y=df.EMA100, name="EMA 100",
        line=dict(color="#e9c46a", width=2), opacity=.9,
    ), row=1, col=1)

    # EMA 200
    fig.add_trace(go.Scatter(
        x=df.index, y=df.EMA200, name="EMA 200",
        line=dict(color="#264653", width=3), opacity=.95,
    ), row=1, col=1)

    # BB shading
    fig.add_trace(go.Scatter(
        x=df.index, y=df.BB_up, name="BB Upper",
        line=dict(color="#4a5568", dash="dot", width=1),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df.BB_lo, name="BB Lower",
        line=dict(color="#4a5568", dash="dot", width=1),
        fill="tonexty", fillcolor="rgba(74,85,104,0.08)",
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df.RSI, name="RSI",
        line=dict(color="#a78bfa", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_color="#e76f51", line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_color="#2a9d8f", line_dash="dot", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=df.MACD, name="MACD",
        line=dict(color="#60a5fa", width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df.MACD_Sig, name="Signal",
        line=dict(color="#f97316", width=1.5),
    ), row=3, col=1)
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in df.MACD_Hist]
    fig.add_trace(go.Bar(
        x=df.index, y=df.MACD_Hist, name="Histogram",
        marker_color=colors, opacity=0.7,
    ), row=3, col=1)

    fig.update_layout(
        height=750, template="plotly_dark",
        paper_bgcolor="#080c12", plot_bgcolor="#080c12",
        xaxis_rangeslider_visible=False,
        showlegend=True, hovermode="x unified",
        font=dict(family="DM Sans", color="#a0b4c8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    )
    fig.update_yaxes(gridcolor="#1e2d3d", showgrid=True)
    fig.update_xaxes(gridcolor="#1e2d3d", showgrid=False)

    st.plotly_chart(fig, use_container_width=True)


def render_prediction(data: dict):
    st.subheader("🤖 AI Price Prediction")
    p15 = predict(add_indicators(data["15m"]), "15m")
    p1h = predict(add_indicators(data["1h"]),  "1h")

    c1, c2 = st.columns(2)
    for col, p, label in [(c1, p15, "Next 15m"), (c2, p1h, "Next 1h")]:
        with col:
            if p:
                arrow_color = "#2a9d8f" if p["move_pct"] > 0 else "#e76f51"
                st.markdown(f"""
                <div class="prediction-box" style="background:linear-gradient(135deg,#0f1923,#131f2e);
                     border:1px solid {arrow_color};border-radius:12px;padding:18px;">
                    <div style="font-size:12px;color:#6b8cad;text-transform:uppercase;">{label}</div>
                    <div style="font-family:'Space Mono';font-size:24px;color:#e8f4f8;margin:8px 0;">
                        ${p['predicted']:,.2f} <span style="color:{arrow_color}">{p['direction']}</span>
                    </div>
                    <div style="font-size:15px;color:{arrow_color};font-weight:600;">
                        {p['move_pct']:+.2f}% ({p['strength']})
                    </div>
                    <div style="font-size:12px;color:#6b8cad;margin-top:6px;">
                        Range ${p['lower']:,.2f} — ${p['upper']:,.2f}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Not enough data for prediction")


def render_backtest(data: dict):
    st.subheader("🧪 Backtest — Prediction Accuracy")
    tabs = st.tabs(["15m", "30m", "1h"])
    for tab, tf in zip(tabs, ["15m", "30m", "1h"]):
        with tab:
            with st.spinner(f"Running {tf} backtest…"):
                res = run_backtest(add_indicators(data[tf]))
            if res is None:
                st.warning("Not enough data")
                continue
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Direction Acc.", f"{res['dir_acc']:.1f}%",   res['grade'])
            c2.metric("Range Acc.",    f"{res['range_acc']:.1f}%")
            c3.metric("Avg Error",     f"{res['mape']:.2f}%")
            c4.metric("Recent (20)",   f"{res['recent_acc']:.1f}%",
                      "Improving" if res['recent_acc'] > res['dir_acc'] else "Declining")

            # chart
            n = len(res['preds'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=res['acts'],  name="Actual",    line=dict(color="#2a9d8f",width=2)))
            fig.add_trace(go.Scatter(y=res['preds'], name="Predicted", line=dict(color="#e9c46a",width=2,dash="dash")))
            fig.update_layout(height=300, template="plotly_dark",
                              paper_bgcolor="#0f1923", plot_bgcolor="#0f1923",
                              title=f"Last {n} predictions vs actual")
            st.plotly_chart(fig, use_container_width=True)

            if res['dir_acc'] >= 65:
                st.success("✅ Strong accuracy — trustworthy signals on this timeframe")
            elif res['dir_acc'] >= 55:
                st.warning("⚠️ Moderate accuracy — trade with confirmation")
            else:
                st.error("❌ Low accuracy on this timeframe — use longer TF")


def render_news():
    st.subheader("📰 Live Crypto & Market News")
    items = get_news()
    c1, c2 = st.columns(2)
    for i, item in enumerate(items[:8]):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f"""
            <div class="news-card">
                <div class="news-title">{item['title']}</div>
                <div class="news-meta">{item['time']} &nbsp;·&nbsp;
                    <a href="{item['link']}" target="_blank"
                       style="color:#2a9d8f;text-decoration:none;">Read →</a>
                </div>
            </div>""", unsafe_allow_html=True)


def render_full_asset(symbol: str):
    """Full single-asset analysis page."""
    data = fetch_data(symbol)
    if data is None:
        st.error(f"Could not load data for **{symbol}**. Check the symbol and try again.")
        return

    # Price metrics
    price    = float(data["15m"]["Close"].iloc[-1])
    prev_day = float(data["1d"]["Close"].iloc[-2]) if len(data["1d"]) > 1 else price
    chg24    = (price - prev_day) / prev_day * 100
    hi24     = float(data["15m"]["High"].tail(96).max())
    lo24     = float(data["15m"]["Low"].tail(96).min())
    vol24    = float(data["15m"]["Volume"].tail(96).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 Price",     f"${price:,.4f}" if price < 5 else f"${price:,.2f}", f"{chg24:+.2f}%")
    m2.metric("📈 24h High",  f"${hi24:,.2f}")
    m3.metric("📉 24h Low",   f"${lo24:,.2f}")
    m4.metric("📦 24h Volume",f"{vol24:,.0f}")

    st.divider()

    # Master signals
    ms = master_signal(data)
    render_master_signals(ms)

    st.divider()

    # Conflict check
    sigs = {}
    for tf in ["15m", "30m", "1h"]:
        sigs[tf] = ema_signal(add_indicators(data[tf]))
    cf = detect_conflicts(data, sigs)
    render_conflict_panel(cf)

    st.divider()

    # TF scanner
    sigs = render_tf_scanner(data)

    st.divider()

    # Trade setups
    render_trade_setups(sigs, rr_ratio, pos_size)

    st.divider()

    # Prediction
    render_prediction(data)

    st.divider()

    # Chart
    render_chart(data, symbol)

    # Backtest (optional)
    if st.session_state.show_bt:
        st.divider()
        render_backtest(data)

    st.divider()

    # News
    render_news()


def render_compact_asset(symbol: str):
    """Compact dual-pane view for multi-asset comparison."""
    data = fetch_data(symbol)
    if data is None:
        st.error(f"No data for {symbol}")
        return

    price = float(data["15m"]["Close"].iloc[-1])
    prev  = float(data["1d"]["Close"].iloc[-2]) if len(data["1d"]) > 1 else price
    chg   = (price - prev) / prev * 100
    pf    = f"${price:,.4f}" if price < 5 else f"${price:,.2f}"
    color = "#2a9d8f" if chg >= 0 else "#e76f51"
    st.markdown(
        f'<div style="font-family:Space Mono;font-size:22px;color:#e8f4f8">{pf} '
        f'<span style="font-size:14px;color:{color}">{chg:+.2f}%</span></div>',
        unsafe_allow_html=True)

    ms = master_signal(data)
    render_master_signals(ms)

    sigs = render_tf_scanner(data)
    st.divider()
    render_trade_setups(sigs, rr_ratio, pos_size)


# ═══════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<h1 style="font-family:'Space Mono';font-size:28px;color:#e8f4f8;margin-bottom:0">
    📊 PRO AI TRADING DESK
</h1>
<p style="color:#4a6a8a;font-size:13px;margin-top:4px">
    EMA 100/200 · 15m Scalping · 30m+1h Intraday · Conflict Detection · Backtest
</p>
""", unsafe_allow_html=True)

# Ticker row
render_ticker_row()
st.divider()

if st.session_state.view == "Single Asset":
    st.markdown(f"### {st.session_state.symbol}")
    render_full_asset(st.session_state.symbol)

else:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {st.session_state.sym1}")
        render_compact_asset(st.session_state.sym1)
    with col2:
        st.markdown(f"### {st.session_state.sym2}")
        render_compact_asset(st.session_state.sym2)

# Auto-refresh
if st.session_state.auto:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
