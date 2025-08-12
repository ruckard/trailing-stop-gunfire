from datetime import datetime, timedelta, timezone
from exchange.btse import fetch_top_symbols_by_volume

def update_symbol_registry(symbols):
    conn = sqlite3.connect(KNOWN_SYMBOLS_DB_PATH)
    c = conn.cursor()

    for sym in symbols:
        c.execute("SELECT status FROM symbols WHERE symbol=?", (sym,))
        row = c.fetchone()
        if not row:
            # new symbol → status = 'new'
            c.execute("INSERT INTO symbols (symbol, status) VALUES (?, ?)", (sym, "new"))

    conn.commit()
    conn.close()

def set_symbol_as_ready(symbol):
    conn = sqlite3.connect(KNOWN_SYMBOLS_DB_PATH)
    c = conn.cursor()

    # Check if the symbol already exists
    c.execute("SELECT 1 FROM symbols WHERE symbol=?", (symbol,))
    exists = c.fetchone() is not None

    if exists:
        c.execute("UPDATE symbols SET status=? WHERE symbol=?", ("ready", symbol))
    else:
        c.execute("INSERT INTO symbols (symbol, status) VALUES (?, ?)", (symbol, "ready"))

    conn.commit()
    conn.close()

def setup_symbol_modes():
    new_symbols = get_new_symbols()

    for sym in new_symbols:
        update_symbol_settings(sym)  # run the actual setup
        print_with_date(f"[SETUP] {sym}: Setting up trading mode → status = 'ready'")
        set_symbol_as_ready(sym)

def filter_old_symbols(summary_data, MIN_CONTRACT_AGE_DAYS):
    cutoff = datetime.now(timezone.utc) - timedelta(days=MIN_CONTRACT_AGE_DAYS)
    eligible = []
    for entry in summary_data:
        contract_start = datetime.fromtimestamp(entry.get("contractStart", 0) / 1000, tz=timezone.utc)
        if contract_start <= cutoff:
            eligible.append(entry["symbol"])
    return eligible

def filter_symbols_by_age_and_volume(market_summary, MIN_CONTRACT_AGE_DAYS, TOP_SYMBOLS_BY_VOLUME):
    # Filter symbols older than MIN_CONTRACT_AGE_DAYS
    aged_symbols = filter_old_symbols(market_summary, MIN_CONTRACT_AGE_DAYS)  # list of strings
    aged_symbol_names = set(aged_symbols)

    # Fetch top volume symbols (no filtering parameter)
    top_symbols = fetch_top_symbols_by_volume(limit=TOP_SYMBOLS_BY_VOLUME)

    # Keep only aged symbols from the top volume list
    filtered_top_symbols = [s for s in top_symbols if s in aged_symbol_names]

    # Add forced additional symbols
    combined = filtered_top_symbols + ADDITIONAL_SYMBOLS

    # Remove excluded and deduplicate
    seen = set()
    final = []
    for s in combined:
        if s not in EXCLUDED_SYMBOLS and s not in seen:
            final.append(s)
            seen.add(s)

    return final

def filter_symbols_by_rank(symbols, long_top_number=3, short_top_number=3, rank_type='EASY6',
                           vol_bottom_percentile=None, vol_top_percentile=None):
    """
    Rank and filter symbols based on trend score and normalized ATR%.
    Skips symbols with 0.0 score and ensures longs are positive slopes, shorts are negative.
    """

    if vol_bottom_percentile is None:
        vol_bottom_percentile = VOL_BOTTOM_PERCENTILE
    if vol_top_percentile is None:
        vol_top_percentile = VOL_TOP_PERCENTILE

    trend_scores = {}
    atr_percents = {}

    for symbol in symbols:
        # Compute score based on rank type
        if rank_type == 'EMA':
            score = calculate_ema_trend_score(symbol)
        elif rank_type == 'TRENDEST':
            score = calculate_trendest_with_rsi(symbol)
        elif rank_type == 'EASY':
            score = calculate_easy_trend_with_rsi(symbol)
        elif rank_type == 'EASY2':
            score = calculate_easy_trend2_with_rsi(symbol)
        elif rank_type == 'EASY3':
            score = calculate_easy_trend3_with_rsi(symbol)
        elif rank_type == 'EASY4':
            score = calculate_easy_trend4_with_rsi(symbol)
        elif rank_type == 'EASY5':
            score = calculate_easy_trend5_with_rsi(symbol)
        elif rank_type == 'EASY6':
            score = calculate_easy_trend6_with_rsi(symbol)
        else:
            raise ValueError(f"Unsupported rank_type: {rank_type}")

        # 🔹 Skip symbols with neutral trend
        if score == 0.0:
            continue

        trend_scores[symbol] = score

        atr = get_atr(symbol)
        price = get_current_price(symbol)
        atr_percent = (Decimal(str(atr)) / Decimal(str(price))) * Decimal("100")
        atr_percents[symbol] = atr_percent

    # 🔹 If no symbols survived, return None
    if not trend_scores:
        print_with_date("[SYMBOLS] No valid symbols after scoring. Returning None.")
        return None, None, None

    # Compute ATR percentiles
    atr_values = [float(v) for v in atr_percents.values()]
    low_cut = np.percentile(atr_values, vol_bottom_percentile)
    high_cut = np.percentile(atr_values, vol_top_percentile)

    # ✅ Instead of filtering, just use all symbols
    filtered_symbols = list(trend_scores.keys())

    if not filtered_symbols:
        print_with_date("[SYMBOLS] No symbols within ATR percentile range. Returning None.")
        return None, None, None

    # Normalize ATR values
    max_atr = max(float(atr_percents[s]) for s in filtered_symbols)
    adjusted_scores = {
        s: float(trend_scores[s]) * (float(atr_percents[s]) / max_atr)
        for s in filtered_symbols
    }

    # Sort symbols by adjusted score
    sorted_symbols = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)

    # 🔹 Ensure longs are positive and shorts are negative slopes
    long_symbols = [s for s, score in sorted_symbols if score > 0][:long_top_number]
    short_symbols = [s for s, score in sorted(adjusted_scores.items(), key=lambda x: x[1]) if score < 0][:short_top_number]

    final_symbols = long_symbols + short_symbols
    print_with_date(f"[SYMBOLS] ATR cut: low={low_cut:.4f}, high={high_cut:.4f}")
    print_with_date(f"[SYMBOLS] Selected LONG: {long_symbols} | SHORT: {short_symbols}")

    return final_symbols, long_symbols, short_symbols
