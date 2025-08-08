"""
Streamlit: Crypto Instrument Analyzer (Cloud-friendly)

Zmiany vs poprzednia wersja:
- Usunięto Binance (451 w PL). Nowa kolejność źródeł: **Yahoo → Coinbase → Bitfinex** (wszystko 5m/24h).
- Normalizacja symbolu: przyjmujemy `BTCUSDT`, `BTC-USD`, `BTCUSD` itd. i mapujemy na odpowiednie formaty dla każdego źródła.
- Lepsze komunikaty błędów.
- Obsługa `st.secrets` (ETHERSCAN_API_KEY, NEWS_API_KEY).

Wymagania: streamlit, pandas, numpy, requests, plotly, yfinance
"""

import math
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
CG_BASE = "https://api.coingecko.com/api/v3"

# ----------------------------
# Helpers: symbol normalization
# ----------------------------

def normalize_symbol(user_input: str) -> dict:
    """Zwraca mapping symboli pod poszczególne źródła danych.
    Akceptujemy formaty: BTCUSDT / BTC-USD / BTCUSD / btcusdt / etc.
    """
    s = (user_input or "").strip().upper()
    # usuwamy separatory
    s_simple = s.replace("/", "").replace("-", "")

    # heurystyki rozpoznania bazowej monety
    mapping = {
        "BTC": "BTC",
        "ETH": "ETH",
        "SOL": "SOL",
        "ADA": "ADA",
        "XRP": "XRP",
        "DOGE": "DOGE",
        "AVAX": "AVAX",
        "BNB": "BNB",
        "DOT": "DOT",
        "LTC": "LTC",
        "LINK": "LINK",
        "ATOM": "ATOM",
        "MATIC": "MATIC",
        "ARB": "ARB",
        "OP": "OP",
    }

    base = None
    for k in mapping.keys():
        if s_simple.startswith(k):
            base = k
            break

    # domyślnie Bitcoin
    if not base:
        base = s_simple  # spróbujemy użyć bez mapy

    # Yahoo / Coinbase: BASE-USD
    yahoo = f"{base}-USD"
    coinbase = f"{base}-USD"

    # Bitfinex: tBASEUSD (bez myślnika)
    bitfinex = f"t{base}USD"

    return {"yahoo": yahoo, "coinbase": coinbase, "bitfinex": bitfinex}

# ----------------------------
# Data sources (5m, 24h)
# ----------------------------

@st.cache_data(show_spinner=False)
def yahoo_klines(ticker: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
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
def coinbase_klines(product: str, granularity: int = 300, limit: int = 300) -> pd.DataFrame:
    """Coinbase public API.
    GET https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300
    Response rows: [time, low, high, open, close, volume] (time in seconds), najnowsze -> najstarsze.
    """
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    headers = {"User-Agent": "streamlit-app/1.0"}
    r = requests.get(url, params={"granularity": granularity}, headers=headers, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Coinbase error: {r.status_code} {r.text[:200]}")
    data = r.json()
    if not data:
        raise RuntimeError("Coinbase zwrócił pusty wynik.")
    # Kolumny: time, low, high, open, close, volume
    cols = ["time","low","high","open","close","volume"]
    df = pd.DataFrame(data, columns=cols)
    # Najnowsze -> najstarsze, odwracamy
    df = df.sort_values("time")
    df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["close_time"] = df["open_time"] + pd.to_timedelta(5, unit="m")
    df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}, inplace=True)
    df = df[["open_time","open","high","low","close","volume","close_time"]]
    df["trades"] = np.nan
    return df

@st.cache_data(show_spinner=False)
def bitfinex_klines(symbol: str, limit: int = 288) -> pd.DataFrame:
    """Bitfinex public API.
    GET https://api-pub.bitfinex.com/v2/candles/trade:5m:tBTCUSD/hist?limit=288&sort=1
    Response rows: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
    """
    url = f"https://api-pub.bitfinex.com/v2/candles/trade:5m:{symbol}/hist"
    headers = {"User-Agent": "streamlit-app/1.0"}
    r = requests.get(url, params={"limit": limit, "sort": 1}, headers=headers, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Bitfinex error: {r.status_code} {r.text[:200]}")
    data = r.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError("Bitfinex zwrócił pusty wynik.")
    cols = ["mts","open","close","high","low","volume"]
    df = pd.DataFrame(data, columns=cols)
    df = df.sort_values("mts")
    df["open_time"] = pd.to_datetime(df["mts"], unit="ms", utc=True)
    # Bitfinex nie daje close_time — zakładamy 5m
    df["close_time"] = df["open_time"] + pd.to_timedelta(5, unit="m")
    # Ujednolicamy kolejność OHLC
    df = df[["open_time","open","high","low","close","volume","close_time"]]
    df["trades"] = np.nan
    return df

# ----------------------------
# Indicators & anomalies
# ----------------------------

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

    out = {
        "price": float(close.iloc[-1]),
        "rsi": float(rsi14.iloc[-1]),
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "sma20": float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else None,
        "vol_anomaly": bool(vol_anom.iloc[-1]) if len(vol_anom) else False,
    }

    score = 0
    reasons = []
    if out["sma20"] and close.iloc[-1] > out["sma20"]:
        score += 1; reasons.append("cena > SMA20 (trend krótkoterminowy w górę)")
    if out["rsi"] < 30:
        score += 0.5; reasons.append("RSI < 30 (wyprzedanie)")
    elif 50 <= out["rsi"] <= 70:
        score += 0.5; reasons.append("RSI 50–70 (momentum dodatnie)")
    elif out["rsi"] > 70:
        score -= 0.5; reasons.append("RSI > 70 (przegrzanie)")

    if out["macd_hist"] > 0 and out["macd"] > out["macd_signal"]:
        score += 1; reasons.append("MACD > sygnału i histogram dodatni")
    if out["vol_anomaly"]:
        score += 0.5; reasons.append("Anomalia wolumenu (możliwy ruch wybiciowy)")

    if score >= 2:
        rec = "Wejście (LONG) – przewaga sygnałów byczych"
    elif score <= -1:
        rec = "Wyjście / SHORT – przewaga sygnałów niedźwiedzich"
    else:
        rec = "Neutralnie / Poczekaj na potwierdzenie"

    confidence = min(max((score + 1) / 3, 0), 1)

    out.update({
        "score": round(score, 2),
        "confidence": round(confidence, 2),
        "reasons": reasons,
    })
    return out

# ----------------------------
# Charts
# ----------------------------

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
    anom_points = df.loc[anomalies] if isinstance(anomalies, pd.Series) else pd.DataFrame()
    if not anom_points.empty:
        fig.add_trace(go.Scatter(
            x=anom_points["open_time"],
            y=anom_points["volume"],
            mode="markers",
            marker=dict(size=9, symbol="x"),
            name="Anomalie"
        ))
    fig.update_layout(title="Wolumen (24h) + Anomalie", xaxis_title="Czas (UTC)", yaxis_title="Wolumen")
    return fig

# ----------------------------
# Coingecko helpers
# ----------------------------

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

@st.cache_data(show_spinner=False)
def try_map_symbol_to_coingecko_id(symbol_or_name: str) -> str | None:
    s = coingecko_search(symbol_or_name)
    if not s:
        return None
    for t in s.get("coins", []):
        if t.get("symbol", "").lower() == symbol_or_name.lower().replace("-usd",""):
            return t.get("id")
        if t.get("name", "").lower() == symbol_or_name.lower():
            return t.get("id")
    coins = s.get("coins", [])
    if coins:
        return coins[0].get("id")
    return None

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Crypto Analyzer 24h", layout="wide")

st.title("🔍 Crypto Analyzer – 24h Insight")
st.caption("Szybka analiza instrumentu krypto: cena, wolumen, newsy, on-chain (opcjonalnie) i sygnał wejścia/wyjścia.")

with st.sidebar:
    st.header("⚙️ Ustawienia / API Keys (opcjonalnie)")
    st.write("Możesz dodać klucze w **App settings → Secrets** na Streamlit Community Cloud.")

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
    st.write("Źródło świec: Yahoo → Coinbase → Bitfinex (automatyczny fallback).")

# Ekran 1: Input symbolu
st.subheader("Krok 1: Podaj kod instrumentu (np. BTCUSDT, ETHUSDT, SOLUSDT lub BTC-USD)")
col1, col2 = st.columns([2,1])
with col1:
    symbol_in = st.text_input("Symbol / Ticker", value="BTCUSDT").strip()
with col2:
    run_btn = st.button("Analizuj", type="primary")

if not run_btn:
    st.info("Wpisz symbol i kliknij **Analizuj**. Przykłady: BTCUSDT, ETHUSDT, SOLUSDT. Dla Yahoo: BTC-USD.")
    st.stop()

mapped = normalize_symbol(symbol_in)

# Pobierz dane rynkowe (24h, 5m)
source_used = None
err_msgs = []
try:
    df = yahoo_klines(mapped["yahoo"])  # 24h, 5m
    source_used = f"Yahoo ({mapped['yahoo']})"
except Exception as e:
    err_msgs.append(str(e))
    try:
        df = coinbase_klines(mapped["coinbase"])  # 5m default
        source_used = f"Coinbase ({mapped['coinbase']})"
    except Exception as e2:
        err_msgs.append(str(e2))
        try:
            df = bitfinex_klines(mapped["bitfinex"])  # 5m
            source_used = f"Bitfinex ({mapped['bitfinex']})"
        except Exception as e3:
            err_msgs.append(str(e3))
            st.error("Nie udało się pobrać danych z Yahoo, Coinbase ani Bitfinex.\n\n" + "\n".join(err_msgs))
            st.stop()

# Przytnij do 24h
now_utc = datetime.now(tz=UTC)
from_ts = now_utc - timedelta(hours=24)
df = df[df["open_time"] >= from_ts].copy()
df = df.sort_values("open_time").reset_index(drop=True)

if df.empty:
    st.warning("Brak danych w ostatnich 24h dla tego symbolu.")
    st.stop()

# Wykresy i statystyki
left, right = st.columns([3,2])
with left:
    st.subheader(f"Wykres 24h – {symbol_in} ({source_used})")
    st.plotly_chart(candle_chart(df, title=f"{symbol_in} – 24h ({source_used})"), use_container_width=True)
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
        "Źródło": source_used,
    }
    st.table(pd.DataFrame(stats.items(), columns=["Metryka", "Wartość"]))

# Sygnały i rekomendacja
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
    if sig['score'] >= 2:
        st.success("; ".join(sig['reasons']))
    elif sig['score'] <= -1:
        st.warning("; ".join(sig['reasons']))
    else:
        st.info("; ".join(sig['reasons']))

# Istotne informacje z sieci (Coingecko + status updates)
with st.expander("🌐 Istotne informacje z sieci (Coingecko)", expanded=False):
    cg_id = try_map_symbol_to_coingecko_id(symbol_in)
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

# On-chain (opcjonalnie)
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

# Podsumowanie / disclaimer
with st.expander("📝 Podsumowanie i uwagi", expanded=False):
    st.markdown(
        """
        **Uwaga:** To narzędzie nie jest poradą inwestycyjną. Algorytm rekomendacji jest heurystyczny i uproszczony.

        **Wskazówki:**
        - Używaj formatów: `BTCUSDT`, `BTC-USD` lub `BTCUSD` (reszta mapuje się automatycznie).
        - Jeśli Yahoo nie zwróci danych, aplikacja użyje Coinbase, a potem Bitfinex.
        - Limity i rate‑limiting mogą powodować chwilowe błędy; spróbuj ponownie.
        """
    )
