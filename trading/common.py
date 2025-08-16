from exchange import btse as exchange
from decimal import Decimal
import state
from utils import print_with_date, debug

def bool_to_int(value: bool) -> int:
    return 1 if value else 0

def int_to_bool(value: int) -> bool:
    return bool(value)

def debug_latest_trades(symbol, limit=10):
    try:
        url_path = '/api/v2.2/user/trade_history'
        url = BASE_URL + url_path
        nonce = str(int(time.time() * 1000))
        sig = generate_signature(API_SECRET, url_path, nonce, "")
        headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': sig,
            'Content-Type': 'application/json'
        }
        params = {
            'symbol': symbol,
            'includeOld': 'true',
            'count': limit  # Get latest N trades
        }
        response = throttled_request("GET", url, headers=headers, params=params)
        response.raise_for_status()
        trades = response.json()

        print_with_date(f"[DEBUG] Showing latest {limit} trades:")
        for i, trade in enumerate(trades, 1):
            print_with_date(
                f"{i}. Trade={trade}"
            )
        return trades

    except Exception as e:
        print_with_date(f"[ERROR] Failed to fetch trade history: {e}")
        return []

def compute_contracts_from_prices(symbols, contract_sizes):
    prices = {}
    notional_per_contract = {}
    per_contract_losses = {}

    for symbol in symbols:
        price = exchange.get_current_price(symbol)
        if price is None or symbol not in contract_sizes:
            continue
        price = Decimal(str(price))
        size = contract_sizes[symbol]
        notional = price * size
        trail_percent = Decimal(str(state.TRAILING_STOPS_MAP.get(symbol, [1])[0])) / Decimal("100")

        prices[symbol] = price
        notional_per_contract[symbol] = notional
        per_contract_losses[symbol] = notional * trail_percent

    if not per_contract_losses:
        return {}, Decimal("0")

    available_usdt = exchange.get_available_balance("USDT")
    target_budget = Decimal(str(available_usdt)) * Decimal("0.8")

    # Start with 1 contract for each symbol
    contracts_map = {sym: Decimal("1") for sym in per_contract_losses}

    def total_notional():
        return sum(contracts_map[sym] * notional_per_contract[sym] for sym in contracts_map)

    def total_loss(sym):
        return per_contract_losses[sym] * contracts_map[sym]

    # Iteratively increase smallest TotalLoss until we reach the budget
    while True:
        current_total_notional = total_notional()
        if current_total_notional >= target_budget:
            break

        # Find symbol with smallest TotalLoss
        symbol_to_increase = min(contracts_map.keys(), key=lambda s: total_loss(s))

        # Check if adding one more contract would exceed the budget
        projected_notional = current_total_notional + notional_per_contract[symbol_to_increase]
        if projected_notional > target_budget:
            break

        # Increase contracts for that symbol
        contracts_map[symbol_to_increase] += 1

    max_expected_loss = max(total_loss(sym) for sym in contracts_map)

    # Convert to int and log
    final_contracts_map = {}
    for symbol in contracts_map:
        c = int(contracts_map[symbol])
        final_contracts_map[symbol] = c
        print_with_date(
            f"[SIZING] {symbol}: Price={prices[symbol]}, Notional/Contract={notional_per_contract[symbol]}, "
            f"PerContractLoss={per_contract_losses[symbol]}, Contracts={c}, "
            f"TotalNotional={notional_per_contract[symbol] * c}, TotalLoss={per_contract_losses[symbol] * c}"
        )

    print_with_date(f"[SIZING] Final TotalNotional={total_notional()}, TargetBudget={target_budget}")
    return final_contracts_map, max_expected_loss

