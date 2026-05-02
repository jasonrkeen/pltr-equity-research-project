from datetime import datetime

# =========================
# Core Asset Configuration
# =========================
TICKER = "PLTR"
START = "2020-09-30"  # PLTR IPO date
END = datetime.today().strftime("%Y-%m-%d")

# =========================
# Benchmarks
# =========================
BENCHMARKS = ["SPY", "XLK", "ITA"]

# =========================
# Peer Group
# =========================
PEERS = ["MSFT", "SNOW", "DDOG", "IBM"]

# =========================
# Multi-Factor Model Proxies
# =========================
FACTOR_ETFS = {
    "market": "SPY",
    "size": "IWM",
    "value": "VLUE",
    "momentum": "MTUM",
    "defense": "ITA",
}

# =========================
# Strategy Parameters
# =========================
MA_SHORT = 50
MA_LONG = 200

# =========================
# Risk-Free Rate
# =========================
RISK_FREE_RATE = 0.04  # fallback if Treasury data is unavailable
RISK_FREE_TICKER = "^IRX"  # 13-week Treasury bill yield proxy

# =========================
# Signal Parameters
# =========================
SIGNAL_RSI_THRESHOLD = 60
SIGNAL_USE_SHORT = False

# =========================
# Enhanced Signal Parameters
# =========================
USE_EMA = True
EMA_SHORT = 20
EMA_LONG = 100

VOL_FILTER_ENABLED = True
VOL_LOOKBACK = 30
VOL_THRESHOLD = 0.05  # minimum daily volatility

# =========================
# Forecast Parameters
# =========================
TRADING_DAYS = 252
NUM_SIMULATIONS = 100

# =========================
# Output Management
# =========================
CLEAN_RUN = True