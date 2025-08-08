# Crypto Analyzer – 24h Insight

Aplikacja Streamlit do analizy instrumentów kryptowalutowych z ostatnich 24h.

## Funkcje
- Pobiera świeczki 5m z Binance (fallback: Yahoo Finance)
- Wyświetla wykres świecowy i wolumenu z anomaliami
- Liczy RSI(14), MACD(12,26,9), SMA20
- Generuje rekomendację wejścia/wyjścia
- Pobiera dane z Coingecko (market, community, dev, status updates)
- Opcjonalna analiza on-chain (ETH, BTC, SOL)

## Uruchomienie lokalne
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy na Streamlit Community Cloud
1. Utwórz repo na GitHub i dodaj pliki: `app.py`, `requirements.txt`, `runtime.txt`, `README.md`
2. Połącz z Streamlit Community Cloud i wybierz branch
3. W **App settings → Secrets** dodaj opcjonalnie:
```
ETHERSCAN_API_KEY = your_key
NEWS_API_KEY = your_key
```
4. Uruchom aplikację.
