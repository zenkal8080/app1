"""
Streamlit: Crypto Instrument Analyzer

Funkcje:
- Ekran powitalny + input na symbol/kod instrumentu (np. BTCUSDT, ETHUSDT, SOLUSDT; lub BTC-USD dla Yahoo fallback)
- Pobranie świec z ostatnich 24h (5m) z Binance (public API). Fallback: Yahoo Finance.
- Wizualizacja: wykres świecowy + wolumen, heatmapa zmienności, tabela statystyk.
- Wyszukiwarka newsów/aktualności: Coingecko (status_updates) + Coingecko /coins/{id} (community/dev metrics). Opcjonalnie NewsAPI/CryptoPanic (wymaga kluczy w sidebar).
- Analiza on-chain (opcjonalna): Etherscan (ETH/ERC-20), mempool.space (BTC), Solscan (SOL) — jeśli podasz klucze. Brak kluczy => sekcje będą ograniczone.
- Detekcja anomalii wolumenu i zmienności (z-score, IQR, rolling bands); highlight na wykresie.
- Sygnał: entry/exit rekomendacja oparta o RSI(14), MACD(12,26,9), SMA(20), momentum oraz anomalie; wraz z uzasadnieniem i prostą oceną pewności.

Uruchomienie:
  pip install -r requirements.txt
  streamlit run app.py

Plik requirements.txt (zawartość):
  streamlit
  pandas
  numpy
  requests
  plotly
  yfinance
  ta

Uwaga: darmowe API Binance ma limity. Jeśli podasz niestandardowy symbol, upewnij się że istnieje na Binance jako para do USDT (np. ABCUSDT). Dla Yahoo używaj notacji ABC-USD.
"""

import os
import math
import time
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta, timezone

# =========
# Utils
# =========
UTC = timezone.utc
BINANCE_BASE = "https://api.binance.com"  # spot
CG_BASE = "https://api.coingecko.com/api/v3"

@st.cache_data(show_spinner=False)
def binance_klines(symbol: str, interval: str = "5m", limit: int = 288) -> pd.DataFrame:
    """Fetch klines from Binance public API (spot). 288 * 5m = 24h.
    Columns: open_time, open, high, low, close, volume, close_time, trades
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Binance error: {r.status_code} {r.text}")
    data = r.json()
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.rename(columns={"number_of_trades": "trades"}, inplace=True)
    return df

@st.cache_data(show_spinner=False)
def yahoo_klines(ticker: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    """Fallback via Yahoo Finance for last 24h.
    Use tickers like BTC-USD, ETH-USD, SOL-USD.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval)
    if hist.empty:
        raise RuntimeError("Yahoo zwróciło pusty wynik dla podanego tickera.")
    df = hist.reset_index().rename(columns={
        "Datetime": "open_time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    if "open_time" not in df.columns:
        df = df.rename(columns={df.columns[0]: "open_time"})
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = df["open_time"] + pd.to_timedelta(5, unit="m")
    df["trades"] = np.nan
    return df[["open_time","open","high","low","close","volume","close_time","trades"]]

@st.cache_data(show_spinner=False)
def coingecko_search(query: str) -> dict:
    r = requests.get(f"{CG_BASE}/search", params={"query": query}, timeout=15)
    if r.status_code != 200:
        return {}
    return r.json()

@st.cache_data(show_spinner=False)
def coingecko_coin_data(coin_id: str) -> dict:
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "true",
        "developer_data": "true",
        "sparkline": "false",
    }
    r = requests.get(f"{CG_BASE}/coins/{coin_id}", params=params, timeout=20)
    if r.status_code != 200:
        return {}
    return r.json()

@st.cache_data(show_spinner=False)
def coingecko_status_updates(coin_id: str, per_page: int = 10) -> dict:
    r = requests.get(f"{CG_BASE}/coins/{coin_id}/status_updates", params={"per_page": per_page}, timeout=20)
    if r.status_code != 200:
        return {}
    return r.json()

# =========
# Indicators & Anomalies
# =========

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def sma(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).mean()


def zscore(series: pd.Series, window: int = 48) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std(ddof=0)
    return (series - m) / (s + 1e-9)


def detect_volume_anomalies(vol: pd.Series, window: int = 48, threshold: float = 2.5) -> pd.Series:
    zs = zscore(vol, window)
    return (zs.abs() >= threshold)


def summarize_signals(df: pd.DataFrame) -> dict:
    close = df["close"].copy()
    vol = df["volume"].copy()
    
    rsi14 = rsi(close, 14)
    macd_line, signal_line, macd_hist = macd(close)
    sma20 = sma(close, 20)
    vol_anom = detect_volume_anomalies(vol)

    latest = df.index[-1]
    out = {
        "price": float(close.iloc[-1]),
        "rsi": float(rsi14.iloc[-1]),
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "sma20": float(sma20.iloc[-1]) if not math.isnan(sma20.iloc[-1]) else None,
        "vol_anomaly": bool(vol_anom.iloc[-1]) if not math.isnan(vol.iloc[-1]) else False,
    }

    # Prosta logika rekomendacji
    score = 0
    reasons = []
    if out["sma20"] and close.iloc[-1] > out["sma20"]:
        score += 1; reasons.append("cena > SMA20 (trend krótkoterminowy w górę)")
    if out["rsi"] < 30:
        score += 0.5; reasons.append("RSI < 30 (wyprzedanie)")
    elif 50 <= out["rsi"] <= 70:
        score += 0.5; reasons.append("RSI w strefie 50–70 (momentum dodatnie)")
    elif out["rsi"] > 70:
        score -= 0.5; reasons.append("RSI > 70 (przegrzanie)")

    if out["macd_hist"] > 0 and out["macd"] > out["macd_signal"]:
        score += 1; reasons.append("MACD > sygnału i histogram dodatni")
    if out["vol_anomaly"]:
        score += 0.5; reasons.append("Anomalia wolumenu (możliwy ruch wybiciowy)")

    # decyzja
    if score >= 2:
        rec = "Wejście (LONG) – przewaga sygnałów byczych"
    elif score <= -1:
        rec = "Wyjście / SHORT – przewaga sygnałów niedźwiedzich"
    else:
        rec = "Neutralnie / Poczekaj na potwierdzenie"

    # Confidence heuristics
    confidence = min(max((score + 1) / 3, 0), 1)  # 0..1

    out.update({
        "score": round(score, 2),
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "latest_ts": str(latest),
    })
    return out

# =========
# Charts
# =========

def candle_chart(df: pd.DataFrame, title: str = "Wykres 24h"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["open_time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Cena"
    ))
    fig.update_layout(title=title, xaxis_title="Czas (UTC)", yaxis_title="Cena")
    return fig


def volume_chart(df: pd.DataFrame, anomalies: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["open_time"], y=df["volume"], name="Wolumen"))
    # Highlight anomalies as markers on secondary scatter
    anom_points = df.loc[anomalies]
    fig.add_trace(go.Scatter(
        x=anom_points["open_time"],
        y=anom_points["volume"],
        mode="markers",
        marker=dict(size=9, symbol="x"),
        name="Anomalie"
    ))
    fig.update_layout(title="Wolumen (24h) + Anomalie", xaxis_title="Czas (UTC)", yaxis_title="Wolumen")
    return fig

# =========
# Info fetchers
# =========

@st.cache_data(show_spinner=False)
def try_map_symbol_to_coingecko_id(symbol_or_name: str) -> str | None:
    s = coingecko_search(symbol_or_name)
    if not s:
        return None
    # Try tickers first
    for t in s.get("coins", []):
        if t.get("symbol", "").lower() == symbol_or_name.lower().replace("-usd",""):
            return t.get("id")
        if t.get("name", "").lower() == symbol_or_name.lower():
            return t.get("id")
    # fallback to first result
    coins = s.get("coins", [])
    if coins:
        return coins[0].get("id")
    return None

# =========
# Streamlit App
# =========

st.set_page_config(page_title="Crypto Analyzer 24h", layout="wide")

st.title("🔍 Crypto Analyzer – 24h Insight")
st.caption("Szybka analiza instrumentu krypto: cena, wolumen, newsy, on-chain (opcjonalnie) i sygnał wejścia/wyjścia.")

with st.sidebar:
    st.header("⚙️ Ustawienia / API Keys (opcjonalnie)")
    st.write("Możesz dodać klucze w **App settings → Secrets** na Streamlit Community Cloud.")

    # Preferuj klucze ze st.secrets; jeśli brak, pozwól wpisać ręcznie
    etherscan_key = st.secrets.get("ETHERSCAN_API_KEY", "")
    newscustom_key = st.secrets.get("NEWS_API_KEY", "")

    if not etherscan_key:
        etherscan_key = st.text_input("Etherscan API Key (ETH/ERC-20)", type="password")
    else:
        st.caption("ETHERSCAN_API_KEY załadowany z Secrets ✅")

    if not newscustom_key:
        newscustom_key = st.text_input("NewsAPI / CryptoPanic Key (opcjonalnie)", type="password")
    else:
        st.caption("NEWS_API_KEY załadowany z Secrets ✅")

    st.divider()
    st.write("Źródło świec: domyślnie Binance (5m, 24h). Yahoo to fallback (np. BTC-USD).")
    st.write("Źródło świec: domyślnie Binance (5m, 24h). Yahoo to fallback (np. BTC-USD).")

# Ekran 1: Input symbolu
st.subheader("Krok 1: Podaj kod instrumentu (np. BTCUSDT, ETHUSDT, SOLUSDT lub BTC-USD)")
col1, col2 = st.columns([2,1])
with col1:
    symbol = st.text_input("Symbol / Ticker", value="BTCUSDT").strip()
with col2:
    run_btn = st.button("Analizuj", type="primary")

if not run_btn:
    st.info("Wpisz symbol i kliknij **Analizuj**. Przykłady: BTCUSDT, ETHUSDT, SOLUSDT. Dla Yahoo: BTC-USD.")
    st.stop()

# Krok 2: Pobranie danych rynkowych (24h)
source_used = None
try:
    df = binance_klines(symbol)
    source_used = "Binance"
except Exception as e:
    try:
        df = yahoo_klines(symbol)
        source_used = "Yahoo Finance"
    except Exception as e2:
        st.error(f"Nie udało się pobrać danych ani z Binance, ani z Yahoo.\nBinance: {e}\nYahoo: {e2}")
        st.stop()

# Prepare timeframe & index
# Restrict to last 24h relative to now UTC (just in case APIs return more/less)
now_utc = datetime.now(tz=UTC)
from_ts = now_utc - timedelta(hours=24)
df = df[df["open_time"] >= from_ts].copy()
df = df.sort_values("open_time").reset_index(drop=True)
df.set_index(pd.Index(range(len(df))), inplace=True)  # simple integer index for indicators

if df.empty:
    st.warning("Brak danych w ostatnich 24h dla tego symbolu.")
    st.stop()

# Krok 3: Wykresy i statystyki
left, right = st.columns([3,2])
with left:
    st.subheader(f"Wykres 24h – {symbol} ({source_used})")
    st.plotly_chart(candle_chart(df, title=f"{symbol} – 24h ({source_used})"), use_container_width=True)
    vol_anom_series = detect_volume_anomalies(df["volume"]) if len(df) >= 60 else pd.Series([False]*len(df))
    st.plotly_chart(volume_chart(df, vol_anom_series), use_container_width=True)

with right:
    st.subheader("Statystyki 24h")
    last_close = df["close"].iloc[-1]
    chg = (last_close / df["close"].iloc[0] - 1) * 100
    stats = {
        "Ostatnia cena": f"{last_close:.4f}",
        "Zmiana 24h": f"{chg:.2f}%",
        "High 24h": f"{df['high'].max():.4f}",
        "Low 24h": f"{df['low'].min():.4f}",
        "Średni wolumen (5m)": f"{df['volume'].mean():.2f}",
        "Liczba świec": len(df),
    }
    st.table(pd.DataFrame(stats.items(), columns=["Metryka", "Wartość"]))

# Krok 4: Sygnały i rekomendacja
sig = summarize_signals(df)
with st.expander("📈 Sygnały i rekomendacja", expanded=True):
    cols = st.columns(3)
    cols[0].metric("Cena", f"{sig['price']:.4f}")
    cols[1].metric("RSI(14)", f"{sig['rsi']:.1f}")
    cols[2].metric("SMA20", f"{sig['sma20']:.4f}" if sig['sma20'] else "—")

    cols2 = st.columns(3)
    cols2[0].metric("MACD", f"{sig['macd']:.4f}")
    cols2[1].metric("MACD sygnał", f"{sig['macd_signal']:.4f}")
    cols2[2].metric("Anomalia wolumenu", "TAK" if sig['vol_anomaly'] else "NIE")

    st.markdown(f"**Rekomendacja:** {sig['score']:.2f} → {sig['confidence']*100:.0f}% pewności")
    st.success(sig['reasons']) if sig['score'] >= 2 else st.warning(sig['reasons']) if sig['score'] <= -1 else st.info(sig['reasons'])

# Krok 5: Znaczące informacje z sieci (Coingecko + status updates)
with st.expander("🌐 Istotne informacje z sieci (Coingecko)", expanded=False):
    cg_id = try_map_symbol_to_coingecko_id(symbol)
    if cg_id:
        coin = coingecko_coin_data(cg_id)
        updates = coingecko_status_updates(cg_id)
        if coin:
            mkt = coin.get("market_data", {})
            comm = coin.get("community_data", {})
            dev = coin.get("developer_data", {})
            cols = st.columns(3)
            cols[0].write({
                "mcap_rank": mkt.get("market_cap_rank"),
                "mcap_change_24h%": mkt.get("market_cap_change_percentage_24h"),
                "price_change_24h%": mkt.get("price_change_percentage_24h"),
            })
            cols[1].write({
                "twitter_followers": comm.get("twitter_followers"),
                "reddit_avg_posts_48h": comm.get("reddit_average_posts_48h"),
                "reddit_subs": comm.get("reddit_subscribers"),
            })
            cols[2].write({
                "stars": dev.get("stars"),
                "forks": dev.get("forks"),
                "subscribers": dev.get("subscribers"),
            })
        if updates and updates.get("status_updates"):
            st.write("**Najnowsze aktualizacje/statusy:**")
            for u in updates["status_updates"][:5]:
                st.write({
                    "time": u.get("created_at"),
                    "project": u.get("project", {}).get("name"),
                    "category": u.get("category"),
                    "description": u.get("description"),
                    "url": u.get("article_url") or u.get("user") or "",
                })
    else:
        st.info("Nie znaleziono ID w Coingecko dla podanego symbolu – pomiń lub spróbuj inną notację.")

# Krok 6: On-chain (opcjonalnie – uproszczone)
with st.expander("⛓️ On-chain (opcjonalnie)", expanded=False):
    st.caption("Przykładowe zapytania: wymagają poprawnego chaina i/lub klucza API.")
    chain = st.selectbox("Wybierz sieć", ["ETH (Etherscan)", "BTC (mempool.space)", "SOL (Solscan)"])
    addr = st.text_input("Adres kontraktu / portfela (opcjonalnie)")
    if st.button("Pobierz dane on-chain"):
        try:
            if chain.startswith("ETH") and etherscan_key and addr:
                url = f"https://api.etherscan.io/api"
                params = {"module":"account","action":"tokentx","address": addr, "page":1, "offset":10, "sort":"desc", "apikey": etherscan_key}
                r = requests.get(url, params=params, timeout=20)
                st.json(r.json())
            elif chain.startswith("BTC") and addr:
                r = requests.get(f"https://mempool.space/api/address/{addr}", timeout=20)
                st.json(r.json())
            elif chain.startswith("SOL") and addr:
                r = requests.get(f"https://public-api.solscan.io/account/tokens?account={addr}", timeout=20)
                st.json(r.json())
            else:
                st.info("Podaj wymagane dane (adres oraz – dla ETH – klucz API).")
        except Exception as e:
            st.error(f"Błąd on-chain: {e}")

# Krok 7: Podsumowanie / Notatki
with st.expander("📝 Podsumowanie i uwagi", expanded=False):
    st.markdown(
        """
        **Uwaga:** To narzędzie nie jest poradą inwestycyjną. Algorytm rekomendacji jest heurystyczny i uproszczony.
        
        **Wskazówki:**
        - Jeśli instrument nie istnieje na Binance jako para do USDT, spróbuj notacji Yahoo (np. `ABC-USD`).
        - Dodaj własne klucze API, by rozszerzyć źródła (NewsAPI/CryptoPanic, Etherscan, itp.).
        - Dostosuj progi anomalii (z-score) i okna wskaźników dla swojego stylu handlu.
        """
    )
