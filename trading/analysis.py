from utils import print_with_date, debug
from exchange import btse as exchange
from decimal import Decimal

# === Win Check ===
def is_win_from_trade(realized_pnl):
    try:
        return float(realized_pnl) > 0
    except:
        return False

def is_breakeven_from_trade(symbol, info, closing_price):
    opening_price = info["opening_price"]
    trail_value = info["trail_value"]
    breakeven_percentage = 10

    current_price = exchange.get_current_price(symbol)
    if not current_price:
        print_with_date("[ERROR] Failed to get current price.")
        raise PriceFetchError("[ERROR] Failed to get current price for breakeven check.")

    try:
        trail_value = Decimal(str(trail_value))
        opening_price = Decimal(str(opening_price))
        closing_price = Decimal(str(closing_price))
        breakeven_percentage = Decimal(str(breakeven_percentage))
        one_percent = Decimal(str(0.01))
        
        return abs(closing_price-opening_price) < (breakeven_percentage * one_percent * trail_value)
    except Exception as e:
        print_with_date(f"[ERROR] Exception during breakeven check: {e}")
        return False
