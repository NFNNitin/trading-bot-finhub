import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from scipy import stats
import requests

# Try to import feedparser, use fallback if not available
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro AI Trader Ultimate", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #303030; padding: 20px; border-radius: 10px; margin-bottom: 10px;}
    .bullish {color: #00ff00; font-weight: bold;}
    .bearish {color: #ff4b4b; font-weight: bold;}
    .price-ticker {
        background: linear-gradient(90deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00ff00;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 15px 0;
    }
    .signal-strong-buy {
        background-color: #00ff00;
        color: black;
        padding: 8px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .signal-strong-sell {
        background-color: #ff4b4b;
        color: white;
        padding: 8px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = 'BINANCE:BTCUSDT'
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'Single Asset'
if 'symbol_1' not in st.session_state:
    st.session_state.symbol_1 = 'BINANCE:BTCUSDT'
if 'symbol_2' not in st.session_state:
    st.session_state.symbol_2 = 'OANDA:XAU_USD'
if 'finnhub_api_key' not in st.session_state:
    st.session_state.finnhub_api_key = ''

# ============================================================
# FINNHUB API — SMART ROUTER
# Crypto  → /crypto/candle   (symbol like BINANCE:BTCUSDT)
# Forex   → /forex/candle    (symbol like OANDA:XAU_USD)
# Stock   → /stock/candle    (symbol like AAPL)
# Quote   → /quote           (stocks) | /crypto/profile2 (crypto)
# ============================================================

# Symbols the live-ticker will display and the correct Finnhub format
TICKER_SYMBOLS = {
    'Bitcoin':      {'finnhub': 'BINANCE:BTCUSDT',  'type': 'crypto'},
    'Gold':         {'finnhub': 'OANDA:XAU_USD',     'type': 'forex'},
    'Silver':       {'finnhub': 'OANDA:XAG_USD',     'type': 'forex'},
    'Dollar Idx':   {'finnhub': 'OANDA:USD_CHF',     'type': 'forex'},
    'XRP':          {'finnhub': 'BINANCE:XRPUSDT',   'type': 'crypto'},
}

def _symbol_type(symbol: str) -> str:
    """Detect whether a symbol is crypto, forex, or stock."""
    s = symbol.upper()
    if ':' in s:
        exchange = s.split(':')[0]
        forex_exchanges = {'OANDA', 'FXCM', 'FX', 'FOREXCOM', 'IC MARKETS'}
        if exchange in forex_exchanges:
            return 'forex'
        return 'crypto'          # BINANCE, COINBASE, KRAKEN, etc.
    return 'stock'               # bare tickers like AAPL, TSLA

def _candle_url(sym_type: str) -> str:
    if sym_type == 'forex':
        return 'https://finnhub.io/api/v1/forex/candle'
    elif sym_type == 'stock':
        return 'https://finnhub.io/api/v1/stock/candle'
    else:
        return 'https://finnhub.io/api/v1/crypto/candle'

def _finnhub_get(url, params, timeout=12):
    """Wrapper — returns (data_dict | None, error_str | None)."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 401:
            return None, "❌ Invalid API key — check the key you entered."
        if r.status_code == 429:
            return None, "⏳ Rate limit hit — wait 60 s then refresh."
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:120]}"
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "🔌 Network error — cannot reach finnhub.io"
    except Exception as e:
        return None, str(e)

def get_finnhub_candles(symbol, resolution, from_ts, to_ts, api_key):
    """
    Fetch OHLCV candles from the correct Finnhub endpoint.
    resolution: '15', '30', '60'  (minutes) or 'D'
    Returns a DataFrame or None.
    """
    sym_type = _symbol_type(symbol)
    url      = _candle_url(sym_type)

    params = {
        'symbol':     symbol,
        'resolution': resolution,
        'from':       int(from_ts),
        'to':         int(to_ts),
        'token':      api_key,
    }

    data, err = _finnhub_get(url, params)
    if err:
        return None, err

    status = data.get('s', 'no_data')
    if status != 'ok':
        hint = ""
        if status == 'no_data':
            hint = " (market may be closed or symbol unsupported on free tier)"
        return None, f"Finnhub returned '{status}'{hint} for {symbol}"

    df = pd.DataFrame({
        'Open':   data['o'],
        'High':   data['h'],
        'Low':    data['l'],
        'Close':  data['c'],
        'Volume': data['v'],
    }, index=pd.to_datetime(data['t'], unit='s'))
    df.index.name = 'timestamp'
    return df.dropna(), None

def get_data_finnhub(symbol, api_key):
    """
    Fetch 15 m, 30 m, and 1 h candles for the given symbol.
    Shows granular error messages so the user knows exactly what went wrong.
    """
    if not api_key:
        st.error("Please enter your Finnhub API key in the sidebar.")
        return None

    now      = int(datetime.utcnow().timestamp())
    windows  = {
        '15m': (now - 5  * 86400, '15'),   # 5 days
        '30m': (now - 7  * 86400, '30'),   # 7 days
        '1h':  (now - 14 * 86400, '60'),   # 14 days
    }

    data   = {}
    errors = {}

    for tf, (from_ts, res) in windows.items():
        df, err = get_finnhub_candles(symbol, res, from_ts, now, api_key)
        if err:
            errors[tf] = err
        elif df is None or len(df) < 10:
            errors[tf] = f"Too few candles returned ({0 if df is None else len(df)})"
        else:
            data[tf] = df

    if errors:
        st.error("**Finnhub data errors:**")
        for tf, msg in errors.items():
            st.error(f"• `{tf}`: {msg}")
        if not data:
            st.info("""
**Common fixes:**
- **Crypto:** Use `BINANCE:BTCUSDT`, `BINANCE:ETHUSDT`, `BINANCE:XRPUSDT`
- **Gold / Silver:** Use `OANDA:XAU_USD` or `OANDA:XAG_USD`
- **Forex:** Use `OANDA:EUR_USD`, `OANDA:GBP_USD`
- **Stocks:** Just the ticker, e.g. `AAPL`, `TSLA`
- The free Finnhub plan supports crypto candles and basic forex candles.
            """)
            return None
        # partial data — still try to render
        st.warning(f"Partial data: missing {list(errors.keys())}. Results may be limited.")

    if len(data) < 2:
        return None
    return data

# --- LIVE PRICE FEED ---
def get_live_quote_finnhub(symbol, sym_type, api_key):
    """
    Return {price, change} for any asset type.
    Stocks   → /quote
    Crypto   → /crypto/candle D (last two daily bars give % change)
    Forex    → /forex/candle  D
    """
    if sym_type == 'stock':
        url = 'https://finnhub.io/api/v1/quote'
        d, _ = _finnhub_get(url, {'symbol': symbol, 'token': api_key})
        if d and d.get('c'):
            price = d['c']
            prev  = d.get('pc', price)
            chg   = ((price - prev) / prev * 100) if prev else 0
            return {'price': price, 'change': chg}

    else:  # crypto or forex
        cand_url = _candle_url(sym_type)
        now  = int(datetime.utcnow().timestamp())
        from_ts = now - 3 * 86400
        d, _ = _finnhub_get(cand_url, {
            'symbol': symbol, 'resolution': 'D',
            'from': from_ts, 'to': now, 'token': api_key
        })
        if d and d.get('s') == 'ok' and len(d.get('c', [])) >= 2:
            price = d['c'][-1]
            prev  = d['c'][-2]
            chg   = ((price - prev) / prev * 100) if prev else 0
            return {'price': price, 'change': chg}

    return {'price': 0, 'change': 0}

@st.cache_data(ttl=60)
def get_live_prices(api_key):
    """Fetch ticker prices for the five headline assets."""
    prices = {}
    for name, meta in TICKER_SYMBOLS.items():
        q = get_live_quote_finnhub(meta['finnhub'], meta['type'], api_key)
        prices[name] = {**q, 'symbol': meta['finnhub']}
    return prices

# --- SIMPLIFIED TECHNICAL INDICATORS (EMA 100 & 200 ONLY) ---
def add_indicators(df):
    """Add ONLY EMA 100 and EMA 200 plus essential indicators"""
    if len(df) < 200:
        return df
    
    # ONLY these two EMAs
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # RSI (keep for momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (keep for crossovers)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # ATR (for stop loss)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # ADX (trend strength)
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.concat([df['High'] - df['Low'], 
                    abs(df['High'] - df['Close'].shift()), 
                    abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
    
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(window=14).mean()
    
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-9)
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    return df

# --- SIMPLIFIED SIGNAL GENERATION (EMA 100/200 FOCUSED) ---
def generate_signal_simplified(df, timeframe_name):
    """
    Simplified signal based ONLY on EMA 100/200 relationship
    Plus RSI, MACD, and ADX for confirmation
    """
    if len(df) < 200:
        return None
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    max_score = 0
    signals = []
    
    # 1. PRICE vs EMA 200 (Most important - 40 points)
    max_score += 40
    if curr['Close'] > curr['EMA200']:
        score += 40
        signals.append("✅ Price above 200 EMA (Major Uptrend)")
    else:
        score -= 40
        signals.append("⛔ Price below 200 EMA (Major Downtrend)")
    
    # 2. PRICE vs EMA 100 (Important - 30 points)
    max_score += 30
    if curr['Close'] > curr['EMA100']:
        score += 30
        signals.append("✅ Price above 100 EMA (Short-term Uptrend)")
    else:
        score -= 30
        signals.append("⛔ Price below 100 EMA (Short-term Downtrend)")
    
    # 3. EMA ALIGNMENT (20 points)
    max_score += 20
    if curr['EMA100'] > curr['EMA200']:
        score += 20
        signals.append("✅ EMA 100 above EMA 200 (Bullish Alignment)")
    else:
        score -= 20
        signals.append("⛔ EMA 100 below EMA 200 (Bearish Alignment)")
    
    # 4. RSI (10 points)
    max_score += 10
    if 40 <= curr['RSI'] <= 60:
        score += 10
        signals.append(f"✅ RSI neutral ({curr['RSI']:.1f})")
    elif curr['RSI'] > 70:
        score -= 5
        signals.append(f"⚠️ RSI Overbought ({curr['RSI']:.1f})")
    elif curr['RSI'] < 30:
        score += 5
        signals.append(f"💎 RSI Oversold ({curr['RSI']:.1f})")
    
    # Calculate normalized score
    normalized_score = ((score + max_score) / (2 * max_score)) * 100
    
    # Determine signal
    if normalized_score >= 75:
        signal_type = "🟢 STRONG BUY"
    elif normalized_score >= 60:
        signal_type = "🟢 BUY"
    elif normalized_score >= 45:
        signal_type = "🟡 NEUTRAL"
    elif normalized_score >= 30:
        signal_type = "🔴 SELL"
    else:
        signal_type = "🔴 STRONG SELL"
    
    return {
        "Signal": signal_type,
        "Score": round(normalized_score, 1),
        "RSI": round(curr['RSI'], 1),
        "MACD": round(curr['MACD'], 4),
        "ADX": round(curr['ADX'], 1),
        "ATR": curr['ATR'],
        "Price": curr['Close'],
        "EMA100": curr['EMA100'],
        "EMA200": curr['EMA200'],
        "Signals": signals
    }

# --- TRADE CALCULATOR ---
def calculate_trade(price, atr, mode="LONG", style="Scalp", risk_reward=1.5):
    """Calculate entry, stop loss, and take profit"""
    multiplier = 1.5 if style == "Scalp" else 2.0
    sl_dist = atr * multiplier
    
    if mode == "LONG":
        sl = price - sl_dist
        tp = price + (sl_dist * risk_reward)
    else:
        sl = price + sl_dist
        tp = price - (sl_dist * risk_reward)
    
    risk_pct = (sl_dist / price) * 100
    reward_pct = abs((tp - price) / price) * 100
    
    return {
        'entry': price,
        'sl': sl,
        'tp': tp,
        'risk_pct': abs(risk_pct),
        'reward_pct': abs(reward_pct),
        'rr_ratio': risk_reward
    }

# --- MASTER SIGNAL (SIMPLIFIED FOR 15m, 30m, 1h) ---
def calculate_master_signal_simplified(data_sets, api_key):
    """
    Simplified master signals:
    - SCALPING: 15m only
    - INTRADAY: 30m + 1h alignment
    """
    
    master_signals = {
        'scalping': {'signal': 'NEUTRAL', 'score': 0, 'reasons': []},
        'intraday': {'signal': 'NEUTRAL', 'score': 0, 'reasons': []}
    }
    
    # Get data
    df_15m = add_indicators(data_sets['15m'])
    df_30m = add_indicators(data_sets['30m'])
    df_1h = add_indicators(data_sets['1h'])
    
    curr_15m = df_15m.iloc[-1]
    curr_30m = df_30m.iloc[-1]
    curr_1h = df_1h.iloc[-1]
    
    # ============ SCALPING (15m) ============
    scalp_score = 0
    scalp_reasons = []
    
    # EMA 100/200 position (60 points)
    if curr_15m['Close'] > curr_15m['EMA200'] and curr_15m['Close'] > curr_15m['EMA100']:
        scalp_score += 60
        scalp_reasons.append("✅ Above both EMAs - Strong bullish")
    elif curr_15m['Close'] < curr_15m['EMA200'] and curr_15m['Close'] < curr_15m['EMA100']:
        scalp_score += 0
        scalp_reasons.append("❌ Below both EMAs - Strong bearish")
    else:
        scalp_score += 30
        scalp_reasons.append("⚠️ Between EMAs - Mixed")
    
    # EMA alignment (20 points)
    if curr_15m['EMA100'] > curr_15m['EMA200']:
        scalp_score += 20
        scalp_reasons.append("✅ EMA 100 > EMA 200")
    
    # RSI (20 points)
    if 45 <= curr_15m['RSI'] <= 65:
        scalp_score += 20
        scalp_reasons.append(f"✅ RSI optimal ({curr_15m['RSI']:.1f})")
    elif curr_15m['RSI'] > 75 or curr_15m['RSI'] < 25:
        scalp_reasons.append(f"⚠️ RSI extreme ({curr_15m['RSI']:.1f})")
    
    # Determine scalping signal
    if scalp_score >= 75:
        master_signals['scalping']['signal'] = "STRONG BUY"
    elif scalp_score >= 60:
        master_signals['scalping']['signal'] = "BUY"
    elif scalp_score <= 35:
        master_signals['scalping']['signal'] = "SELL"
    else:
        master_signals['scalping']['signal'] = "NEUTRAL"
    
    master_signals['scalping']['score'] = scalp_score
    master_signals['scalping']['reasons'] = scalp_reasons
    
    # ============ INTRADAY (30m + 1h) ============
    intra_score = 0
    intra_reasons = []
    
    # 30m EMA position (30 points)
    if curr_30m['Close'] > curr_30m['EMA200']:
        intra_score += 30
        intra_reasons.append("✅ 30m: Above EMA 200")
    else:
        intra_reasons.append("⛔ 30m: Below EMA 200")
    
    # 1h EMA position (40 points - most important)
    if curr_1h['Close'] > curr_1h['EMA200'] and curr_1h['EMA100'] > curr_1h['EMA200']:
        intra_score += 40
        intra_reasons.append("✅ 1h: Perfect EMA alignment")
    elif curr_1h['Close'] > curr_1h['EMA200']:
        intra_score += 25
        intra_reasons.append("✅ 1h: Above EMA 200")
    else:
        intra_reasons.append("⛔ 1h: Below EMA 200")
    
    # ADX (30 points - trend strength)
    if curr_1h['ADX'] > 25:
        intra_score += 30
        intra_reasons.append(f"✅ Strong trend (ADX: {curr_1h['ADX']:.1f})")
    else:
        intra_reasons.append(f"⚠️ Weak trend (ADX: {curr_1h['ADX']:.1f})")
    
    # Determine intraday signal
    if intra_score >= 75:
        master_signals['intraday']['signal'] = "STRONG BUY"
    elif intra_score >= 60:
        master_signals['intraday']['signal'] = "BUY"
    elif intra_score <= 35:
        master_signals['intraday']['signal'] = "SELL"
    else:
        master_signals['intraday']['signal'] = "NEUTRAL"
    
    master_signals['intraday']['score'] = intra_score
    master_signals['intraday']['reasons'] = intra_reasons
    
    return master_signals

# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("⚙️ Settings")

# Finnhub API Key
api_key = st.sidebar.text_input(
    "Finnhub API Key",
    value=st.session_state.finnhub_api_key,
    type="password",
    help="Get free API key at https://finnhub.io"
)
if api_key:
    st.session_state.finnhub_api_key = api_key

st.sidebar.caption("📝 [Get Free Finnhub API Key](https://finnhub.io)")
st.sidebar.divider()

# View mode
view_mode = st.sidebar.radio(
    "📊 View Mode",
    ["Single Asset", "Multi-Asset Comparison"],
    index=0 if st.session_state.view_mode == "Single Asset" else 1
)
st.session_state.view_mode = view_mode

st.sidebar.divider()

# Asset selection
if view_mode == "Single Asset":
    symbol = st.sidebar.text_input(
        "Asset Symbol (Finnhub format)",
        value=st.session_state.current_symbol,
        help="Examples: BINANCE:BTCUSDT, OANDA:XAU_USD"
    ).upper()
    if symbol:
        st.session_state.current_symbol = symbol
else:
    symbol_1 = st.sidebar.text_input(
        "Asset 1",
        value=st.session_state.symbol_1
    ).upper()
    symbol_2 = st.sidebar.text_input(
        "Asset 2",
        value=st.session_state.symbol_2
    ).upper()
    if symbol_1:
        st.session_state.symbol_1 = symbol_1
    if symbol_2:
        st.session_state.symbol_2 = symbol_2

st.sidebar.divider()

# Manual refresh
if st.sidebar.button("🔄 REFRESH NOW", use_container_width=True):
    st.session_state.last_refresh = datetime.now()
    st.rerun()

st.session_state.auto_refresh = st.sidebar.checkbox("Auto-Refresh (60s)", value=st.session_state.auto_refresh)

st.sidebar.divider()

# Quick select
st.sidebar.subheader("⚡ Quick Select")
quick_assets = {
    'BTC': 'BINANCE:BTCUSDT',
    'ETH': 'BINANCE:ETHUSDT',
    'Gold': 'OANDA:XAU_USD',
    'Silver': 'OANDA:XAG_USD',
    'EUR/USD': 'OANDA:EUR_USD',
    'XRP': 'BINANCE:XRPUSDT'
}

cols = st.sidebar.columns(2)
for idx, (name, ticker) in enumerate(quick_assets.items()):
    with cols[idx % 2]:
        if st.button(name, use_container_width=True, key=f"quick_{ticker}"):
            if view_mode == "Single Asset":
                st.session_state.current_symbol = ticker
                st.rerun()
            else:
                st.session_state.symbol_1 = ticker
                st.rerun()

st.sidebar.divider()

# Risk settings
st.sidebar.subheader("Risk Management")
risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 3.0, 1.5, 0.5)
position_size = st.sidebar.number_input("Position Size ($)", min_value=100, value=1000, step=100)

st.sidebar.info(f"Last Refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# ── API Diagnostic ──────────────────────────────────────────
with st.sidebar.expander("🔎 API Diagnostic"):
    if st.button("Test API Key", use_container_width=True):
        if st.session_state.finnhub_api_key:
            with st.spinner("Testing…"):
                d, err = _finnhub_get(
                    'https://finnhub.io/api/v1/crypto/candle',
                    {
                        'symbol': 'BINANCE:BTCUSDT',
                        'resolution': '15',
                        'from': int(datetime.utcnow().timestamp()) - 86400,
                        'to':   int(datetime.utcnow().timestamp()),
                        'token': st.session_state.finnhub_api_key,
                    }
                )
            if err:
                st.error(f"Test FAILED: {err}")
            elif d and d.get('s') == 'ok':
                st.success(f"✅ Key works! Got {len(d['t'])} BTC candles.")
            else:
                st.warning(f"Key OK but got: {d.get('s')}")
        else:
            st.warning("Enter your API key first.")

# ============================================
# MAIN DASHBOARD
# ============================================

st.title("📊 Ultimate AI Trading Dashboard (Finnhub)")
st.caption("🔧 Simplified: EMA 100/200 Only | 15m Scalping | 30m+1h Intraday")

# Check API key
if not st.session_state.finnhub_api_key:
    st.warning("⚠️ Enter your Finnhub API key in the sidebar to continue.")
    st.info("""
**How to get your FREE Finnhub API key:**
1. Go to **[https://finnhub.io](https://finnhub.io)**
2. Click **"Get free API key"** → sign up
3. Copy the key and paste it in the sidebar

---

**✅ Supported symbols (copy-paste ready):**

| Asset | Symbol to enter |
|-------|----------------|
| Bitcoin | `BINANCE:BTCUSDT` |
| Ethereum | `BINANCE:ETHUSDT` |
| XRP | `BINANCE:XRPUSDT` |
| Gold | `OANDA:XAU_USD` |
| Silver | `OANDA:XAG_USD` |
| EUR/USD | `OANDA:EUR_USD` |
| Apple | `AAPL` |
| Tesla | `TSLA` |

**ℹ️ Free tier covers:** crypto candles (Binance) + forex candles (OANDA) + stock quotes.
    """)
    st.stop()

# Live price ticker
st.subheader("🌐 Live Market Feed")
live_prices = get_live_prices(st.session_state.finnhub_api_key)

ticker_cols = st.columns(5)
for idx, (name, data) in enumerate(live_prices.items()):
    with ticker_cols[idx]:
        price = data.get('price', 0)
        change = data.get('change', 0)
        color = "green" if change >= 0 else "red"
        arrow = "▲" if change >= 0 else "▼"
        # Format price: show more decimals for small values (forex)
        price_str = f"${price:,.4f}" if price < 10 else f"${price:,.2f}"
        st.markdown(f"""
        <div class="price-ticker" style="border-left-color: {color};">
            <div style="font-size: 12px; color: #888;">{name}</div>
            <div style="font-size: 18px; font-weight: bold;">{price_str}</div>
            <div style="font-size: 14px; color: {color};">
                {arrow} {abs(change):.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Main analysis
if view_mode == "Single Asset":
    symbol = st.session_state.current_symbol
    st.subheader(f"📈 Analysis: {symbol}")
    
    data_sets = get_data_finnhub(symbol, st.session_state.finnhub_api_key)
    
    if data_sets:
        current_price = data_sets['15m'].iloc[-1]['Close']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Current Price", f"${current_price:,.2f}")
        with col2:
            high_24h = data_sets['15m']['High'].tail(96).max()  # 96 15-min = 24h
            st.metric("📊 24h High", f"${high_24h:,.2f}")
        with col3:
            low_24h = data_sets['15m']['Low'].tail(96).min()
            st.metric("📊 24h Low", f"${low_24h:,.2f}")
        
        st.divider()
        
        # MASTER SIGNALS
        st.subheader("🎯 MASTER SIGNALS")
        master_signals = calculate_master_signal_simplified(data_sets, st.session_state.finnhub_api_key)
        
        sig_col1, sig_col2 = st.columns(2)
        
        # Scalping
        with sig_col1:
            scalp_sig = master_signals['scalping']
            
            if "STRONG BUY" in scalp_sig['signal']:
                bg_color, icon = "#00ff00", "🚀"
            elif "BUY" in scalp_sig['signal']:
                bg_color, icon = "#90EE90", "📈"
            elif "SELL" in scalp_sig['signal']:
                bg_color, icon = "#ff4b4b", "📉"
            else:
                bg_color, icon = "#808080", "⏸️"
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 15px; text-align: center;">
                <h3 style="margin: 0;">⚡ SCALPING (15m)</h3>
                <div style="font-size: 32px; margin: 10px 0;">{icon} {scalp_sig['signal']}</div>
                <div style="font-size: 18px;">Score: {scalp_sig['score']}/100</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📋 See Why"):
                for reason in scalp_sig['reasons']:
                    st.write(reason)
        
        # Intraday
        with sig_col2:
            intra_sig = master_signals['intraday']
            
            if "STRONG BUY" in intra_sig['signal']:
                bg_color, icon = "#00ff00", "🚀"
            elif "BUY" in intra_sig['signal']:
                bg_color, icon = "#90EE90", "📈"
            elif "SELL" in intra_sig['signal']:
                bg_color, icon = "#ff4b4b", "📉"
            else:
                bg_color, icon = "#808080", "⏸️"
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 15px; text-align: center;">
                <h3 style="margin: 0;">📅 INTRADAY (30m+1h)</h3>
                <div style="font-size: 32px; margin: 10px 0;">{icon} {intra_sig['signal']}</div>
                <div style="font-size: 18px;">Score: {intra_sig['score']}/100</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📋 See Why"):
                for reason in intra_sig['reasons']:
                    st.write(reason)
        
        st.divider()
        
        # Timeframe scanner
        st.subheader("⏰ Timeframe Scanner")
        
        tf_cols = st.columns(3)
        timeframes = ['15m', '30m', '1h']
        
        for i, tf in enumerate(timeframes):
            df = add_indicators(data_sets[tf])
            sig = generate_signal_simplified(df, tf)
            
            with tf_cols[i]:
                st.markdown(f"**{tf.upper()}**")
                if sig:
                    if "STRONG BUY" in sig['Signal']:
                        st.markdown(f"<div class='signal-strong-buy'>{sig['Signal']}</div>", unsafe_allow_html=True)
                    elif "STRONG SELL" in sig['Signal']:
                        st.markdown(f"<div class='signal-strong-sell'>{sig['Signal']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{sig['Signal']}**")
                    
                    st.progress(sig['Score'] / 100)
                    st.caption(f"Score: {sig['Score']}/100")
                    st.write(f"RSI: {sig['RSI']}")
                    st.write(f"ADX: {sig['ADX']}")
                    
                    with st.expander("Details"):
                        for s in sig['Signals'][:3]:
                            st.caption(s)
        
        st.divider()
        
        # Trade setups
        st.subheader("🎯 Trade Setups")
        
        setup_cols = st.columns(2)
        
        # Scalping setup
        with setup_cols[0]:
            st.markdown("### ⚡ Scalping (15m)")
            df_15m = add_indicators(data_sets['15m'])
            sig_15m = generate_signal_simplified(df_15m, '15m')
            
            if sig_15m and "BUY" in sig_15m['Signal']:
                trade = calculate_trade(sig_15m['Price'], sig_15m['ATR'], "LONG", "Scalp", risk_reward)
                st.success("📈 LONG SETUP")
                st.write(f"**Entry:** ${trade['entry']:,.2f}")
                st.write(f"🎯 **TP:** ${trade['tp']:,.2f} (+{trade['reward_pct']:.2f}%)")
                st.write(f"🛑 **SL:** ${trade['sl']:,.2f} (-{trade['risk_pct']:.2f}%)")
                
                risk_amount = position_size * (trade['risk_pct'] / 100)
                st.info(f"💰 Risk: ${risk_amount:.2f}")
                
            elif sig_15m and "SELL" in sig_15m['Signal']:
                trade = calculate_trade(sig_15m['Price'], sig_15m['ATR'], "SHORT", "Scalp", risk_reward)
                st.error("📉 SHORT SETUP")
                st.write(f"**Entry:** ${trade['entry']:,.2f}")
                st.write(f"🎯 **TP:** ${trade['tp']:,.2f}")
                st.write(f"🛑 **SL:** ${trade['sl']:,.2f}")
                
                risk_amount = position_size * (trade['risk_pct'] / 100)
                st.info(f"💰 Risk: ${risk_amount:.2f}")
            else:
                st.info("⏸️ No Setup")
        
        # Intraday setup
        with setup_cols[1]:
            st.markdown("### 📅 Intraday (1h)")
            df_1h = add_indicators(data_sets['1h'])
            sig_1h = generate_signal_simplified(df_1h, '1h')
            
            if sig_1h and "BUY" in sig_1h['Signal']:
                trade = calculate_trade(sig_1h['Price'], sig_1h['ATR'], "LONG", "Intraday", risk_reward)
                st.success("📈 LONG SETUP")
                st.write(f"**Entry:** ${trade['entry']:,.2f}")
                st.write(f"🎯 **TP:** ${trade['tp']:,.2f} (+{trade['reward_pct']:.2f}%)")
                st.write(f"🛑 **SL:** ${trade['sl']:,.2f} (-{trade['risk_pct']:.2f}%)")
                
                risk_amount = position_size * (trade['risk_pct'] / 100)
                st.info(f"💰 Risk: ${risk_amount:.2f}")
                
            elif sig_1h and "SELL" in sig_1h['Signal']:
                trade = calculate_trade(sig_1h['Price'], sig_1h['ATR'], "SHORT", "Intraday", risk_reward)
                st.error("📉 SHORT SETUP")
                st.write(f"**Entry:** ${trade['entry']:,.2f}")
                st.write(f"🎯 **TP:** ${trade['tp']:,.2f}")
                st.write(f"🛑 **SL:** ${trade['sl']:,.2f}")
                
                risk_amount = position_size * (trade['risk_pct'] / 100)
                st.info(f"💰 Risk: ${risk_amount:.2f}")
            else:
                st.info("⏸️ No Setup")
        
        st.divider()
        
        # Chart
        st.subheader("📈 Price Chart with EMA 100/200 (1H)")
        
        chart_df = add_indicators(data_sets['1h'])
        
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name="Price"
        ))
        
        # EMA 100
        fig.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df['EMA100'],
            name="EMA 100",
            line=dict(color='orange', width=2)
        ))
        
        # EMA 200
        fig.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df['EMA200'],
            name="EMA 200",
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Failed to fetch data. Please check your API key and symbol format.")

else:
    # Multi-asset view
    st.subheader(f"📊 Multi-Asset: {st.session_state.symbol_1} vs {st.session_state.symbol_2}")
    
    col1, col2 = st.columns(2)
    
    for col, symbol in zip([col1, col2], [st.session_state.symbol_1, st.session_state.symbol_2]):
        with col:
            st.markdown(f"### {symbol}")
            data = get_data_finnhub(symbol, st.session_state.finnhub_api_key)
            
            if data:
                current = data['15m'].iloc[-1]['Close']
                st.metric("Price", f"${current:,.2f}")
                
                master = calculate_master_signal_simplified(data, st.session_state.finnhub_api_key)
                
                st.markdown("**Signals:**")
                st.write(f"⚡ Scalp: {master['scalping']['signal']} ({master['scalping']['score']}/100)")
                st.write(f"📅 Intra: {master['intraday']['signal']} ({master['intraday']['score']}/100)")
            else:
                st.error("Failed to fetch data")

# Auto-refresh
if st.session_state.auto_refresh:
    time.sleep(60)
    st.rerun()
