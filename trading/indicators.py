import pandas as pd
from utils import print_with_date
from exchange.btse import fetch_5m_ohlcv  # temporary — will be configurable later
from decimal import Decimal
from trading.trend import calculate_atr

def get_atr(symbol, period=14):
    try:
        df = fetch_5m_ohlcv(symbol)
        if df is None or df.empty:
            print_with_date(f"[ATR] No candle data for {symbol}")
            return Decimal("0")

        atr_value = calculate_atr(df, ma_period=period)

        # Handle if calculate_atr returns a float instead of Series
        if isinstance(atr_value, (float, int)):
            latest_atr = Decimal(str(atr_value))
        else:
            if atr_value.empty:
                print_with_date(f"[ATR] Could not compute ATR for {symbol}")
                return Decimal("0")
            latest_atr = Decimal(str(atr_value.iloc[-1]))

        return latest_atr

    except Exception as e:
        print_with_date(f"[ATR ERROR] {symbol}: {e}")
        return Decimal("0")
