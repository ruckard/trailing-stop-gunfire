import time
from exchange import btse as exchange
from utils import print_with_date, debug
import numpy as np
import state
import db.positions as positionsdb
from trading.common import get_dynamic_trade_max_candles
import pandas as pd
from datetime import datetime, timedelta

TRENDRANGE_CACHE = {}

ALIVE_CHART_CACHE = {}
ALIVE_FORCE_REFRESH_HOURS = 1           # full recalc required after 1h
ALIVE_MIN_SPACING_MINUTES = 10          # cannot refresh more frequently than this

CHOPPINESS_SCORE_CACHE = {}
CHOPPY_FORCE_REFRESH_HOURS = 1
CHOPPY_MIN_SPACING_MINUTES = 10

def place_trend_positions(symbol, sides):
    # TODO: Maybe improve it so that we don't have a lazy import
    from trading.orders import place_trailing_stop

    # Fetch the previously saved score for this symbol
    trend_score_result = state.TREND_SCORES_TMP.get(symbol)
    if trend_score_result is None:
        print_with_date(f"[WARN] No trend score found for {symbol}. Skipping position placement.")
        return

    trend_score_value = float(trend_score_result)  # this is the score from calculate_easy_trend10_with_rsi

    # Compute dynamic multiplier based on score magnitude
    def score_to_dynamic_multiplier(score: float, scale=800.0) -> float:
        import numpy as np
        return np.tanh(abs(score) * scale)

    score_multiplier = score_to_dynamic_multiplier(trend_score_value)

    # Iterate over trailing callbacks
    for i, callback in enumerate(state.TRAILING_STOPS_MAP[symbol]):
        for side in sides:
            pid = f"{side.lower()}-{i}"

            # Base contracts (capped by 80% total allocation)
            base_contracts = state.CONTRACTS_MAP.get(symbol, 1)

            # Scale dynamically by trend score
            contracts = max(1, int(base_contracts * score_multiplier))

            result = place_trailing_stop(symbol, side, callback, contracts)
            # Check if the result is valid (i.e., position_id and opening_order_id and closing_order_id are returned)

            if result is None or result[0] is None or result[1] is None or result[2] is None:
                print_with_date(f"[ERROR] Failed to place trailing stop for {symbol} {side} at {callback}%")
                continue

            pos_id, opening_order_id, closing_order_id, opening_price, trail_value, score = result
            max_candles = get_dynamic_trade_max_candles(symbol, trend_score_value)

            state.positions[symbol][pid] = {
                "position_id": pos_id,
                "opening_order_id": opening_order_id,
                "closing_order_id": closing_order_id,
                "side": side,
                "callback": callback,
                "active": True,
                "opening_price": opening_price,
                "trail_value": trail_value,
                "opened_at": time.time(),
                "max_candles": max_candles
            }

            position_info = state.positions[symbol][pid]
            positionsdb.update_position(pid, position_info, symbol)

def classify_trend_or_range_real(symbol, lookback=50, threshold=0.0003):
    """
    Classifies symbol as 'trend' or 'range' based on trend strength.
    Returns: 'trend', 'range', or 'unknown'
    """
    try:
        result = calculate_easy_trend10_with_rsi(symbol, lookback=lookback)

        # Extract raw slope if available
        if isinstance(result, dict):
            score_value = result.get("score", 0.0)
            # Use raw_slope if we want threshold comparison to ignore confidence
            raw_slope = result.get("raw_slope", score_value)
        else:
            score_value = result
            raw_slope = result

        # --- CLASSIFICATION BASED ON RAW SLOPE ---
        if raw_slope == 0.0:
            return "range"
        elif abs(raw_slope) >= threshold:
            return "trend"
        else:
            return "range"

    except Exception as e:
        print_with_date(f"[ERROR] Classify failed for {symbol}: {e}")
        return "unknown"

def is_low_volatility_symbol(
    symbol,
    lookback=50,
    volatility_low_cut=1.50,
    trim_fraction=0.1,
    method="trimmed_ewma",          # default: trimmed_ewma
    ewma_span=None,                 # dynamic: set below
    ewma_weighted_mix=0.6           # recommended balance between robustness and recent bias
):
    """
    Determine whether a symbol is low volatility based on body/shadow ratio using
    a trimmed exponential weighted mean (trimmed_ewma) approach by default.

    Returns (is_low: bool, metric_value: float)
    """
    try:
        df = exchange.fetch_5m_ohlcv(symbol)
        if df is None or df.empty or len(df) < lookback:
            debug(f"[WARN] Not enough data for {symbol}")
            return False, 0.0

        df = df.tail(lookback).copy()

        if "open" not in df.columns or "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
            df = pd.DataFrame(df, columns=["timestamp", "open", "high", "low", "close", "volume"])

        df["body_size"] = (df["close"] - df["open"]).abs()
        df["shadow_size"] = (df["high"] - df["low"]) - df["body_size"]
        df = df[df["shadow_size"] > 0]
        if df.empty:
            debug(f"[WARN] No valid candles for {symbol} after filtering shadows")
            return False, 0.0

        ratios = (df["body_size"] / df["shadow_size"]).dropna()
        if ratios.empty:
            debug(f"[WARN] No valid ratio values for {symbol}")
            return False, 0.0

        # Medium robustness trimming
        def trimmed_array(arr, trim_frac):
            arr_sorted = np.sort(arr)
            n = len(arr_sorted)
            k = int(np.floor(n * trim_frac))
            if n - 2 * k <= 0:
                return arr_sorted
            return arr_sorted[k : n - k]

        # Set dynamic EWMA span based on lookback (medium recent bias)
        if ewma_span is None:
            ewma_span = max(3, int(lookback / 10))

        # Compute trimmed mean and ewma components
        trimmed = trimmed_array(ratios.to_numpy(), trim_fraction)
        trimmed_mean_val = float(np.mean(trimmed)) if trimmed.size > 0 else float(ratios.median())
        ewma_val = float(ratios.ewm(span=ewma_span, adjust=False).mean().iloc[-1])

        # Combine both for balanced responsiveness
        metric_value = ewma_weighted_mix * trimmed_mean_val + (1 - ewma_weighted_mix) * ewma_val

        debug(
            f"[INFO] {symbol} volatility metric (trimmed_ewma): {metric_value:.4f} "
            f"(threshold={volatility_low_cut}, lookback={lookback})"
        )

        return metric_value >= volatility_low_cut, float(metric_value)

    except Exception as e:
        debug(f"[ERROR] Failed to calculate volatility for {symbol}: {e}")
        return False, 0.0

def alive_chart_score(
    symbol,
    lookback=60,
    min_range_ratio=0.0002,
    min_volatility_ratio=0.00015,
    min_active_candles=1,
    min_unique_prices=3,
    max_micro_range_ratio=0.0008,
    max_micro_range_fraction=0.95,
    max_flat_run=30,
):
    """
    Returns an alive chart score in range [0.0, 1.0].
    0.0 = effectively dead
    1.0 = very healthy / active market
    """

    now = datetime.utcnow()

    # -----------------------------------------------------------------------
    # CACHE LOOKUP & REFRESH LOGIC
    # -----------------------------------------------------------------------
    cache = ALIVE_CHART_CACHE.get(symbol)

    if cache:
        last_refresh = cache["last_refresh"]

        force_refresh_due = (now - last_refresh) >= timedelta(hours=ALIVE_FORCE_REFRESH_HOURS)
        spacing_block = (now - last_refresh) < timedelta(minutes=ALIVE_MIN_SPACING_MINUTES)

        if not force_refresh_due and spacing_block:
            return cache["alive_score"]
    else:
        ALIVE_CHART_CACHE[symbol] = {
            "alive_score": 0.0,
            "last_refresh": datetime(2000, 1, 1),
        }
        cache = ALIVE_CHART_CACHE[symbol]

    # -----------------------------------------------------------------------
    # FETCH 1-MINUTE DATA
    # -----------------------------------------------------------------------
    try:
        df = exchange.fetch_1m_ohlcv(symbol)
    except Exception:
        ALIVE_CHART_CACHE[symbol]["alive_score"] = 0.0
        ALIVE_CHART_CACHE[symbol]["last_refresh"] = now
        return 0.0

    if df is None or len(df) < lookback:
        ALIVE_CHART_CACHE[symbol]["alive_score"] = 0.0
        ALIVE_CHART_CACHE[symbol]["last_refresh"] = now
        return 0.0

    df = df.tail(lookback)
    opens  = df["open"].astype(float).values
    highs  = df["high"].astype(float).values
    lows   = df["low"].astype(float).values
    closes = df["close"].astype(float).values

    price_mean = np.mean(closes)
    if price_mean <= 0 or np.isnan(price_mean):
        ALIVE_CHART_CACHE[symbol]["alive_score"] = 0.0
        ALIVE_CHART_CACHE[symbol]["last_refresh"] = now
        return 0.0

    # -----------------------------------------------------------------------
    # 1. RANGE SCORE
    # -----------------------------------------------------------------------
    price_range_ratio = (np.max(closes) - np.min(closes)) / price_mean
    range_score = min(1.0, price_range_ratio / min_range_ratio)

    # -----------------------------------------------------------------------
    # 2. VOLATILITY SCORE
    # -----------------------------------------------------------------------
    price_std_ratio = np.std(closes) / price_mean
    vol_score = min(1.0, price_std_ratio / min_volatility_ratio)

    # -----------------------------------------------------------------------
    # 3. ACTIVE CANDLE SCORE
    # -----------------------------------------------------------------------
    bodies = np.abs(closes - opens) / price_mean
    active_count = np.sum(bodies > 0.0003)
    activity_score = min(1.0, active_count / max(1, min_active_candles))

    # -----------------------------------------------------------------------
    # 4. PRICE DIVERSITY SCORE
    # -----------------------------------------------------------------------
    unique_prices = len(set(closes))
    diversity_score = min(1.0, unique_prices / max(1, min_unique_prices))

    # -----------------------------------------------------------------------
    # 5. MICRO-RANGE PENALTY
    # -----------------------------------------------------------------------
    ranges = (highs - lows) / price_mean
    micro_fraction = np.sum(ranges < max_micro_range_ratio) / lookback
    micro_score = max(0.0, 1.0 - (micro_fraction / max_micro_range_fraction))

    # -----------------------------------------------------------------------
    # 6. FLAT-RUN PENALTY
    # -----------------------------------------------------------------------
    flat_run = 0
    worst_flat = 0
    for i in range(1, lookback):
        if opens[i] == closes[i] == closes[i - 1]:
            flat_run += 1
            worst_flat = max(worst_flat, flat_run)
        else:
            flat_run = 0

    flat_score = max(0.0, 1.0 - (worst_flat / max_flat_run))

    # -----------------------------------------------------------------------
    # FINAL ALIVE SCORE
    # -----------------------------------------------------------------------
    alive_score = (
        range_score
        * vol_score
        * activity_score
        * diversity_score
        * micro_score
        * flat_score
    )

    alive_score = float(np.clip(alive_score, 0.0, 1.0))

    ALIVE_CHART_CACHE[symbol]["alive_score"] = alive_score
    ALIVE_CHART_CACHE[symbol]["last_refresh"] = now

    return alive_score

def choppiness_score(
    symbol,
    lookback=60,
    min_efficiency=0.025,
):
    """
    Returns a choppiness score in range [0.0, 1.0].

    0.0 → highly directional / clean trend
    1.0 → extremely choppy / noisy
    """

    now = datetime.utcnow()

    # -----------------------------------------------------------------------
    # CACHE LOOKUP & REFRESH LOGIC
    # -----------------------------------------------------------------------
    cache = CHOPPINESS_SCORE_CACHE.get(symbol)

    if cache:
        last_refresh = cache["last_refresh"]

        force_refresh_due = (now - last_refresh) >= timedelta(hours=CHOPPY_FORCE_REFRESH_HOURS)
        spacing_block = (now - last_refresh) < timedelta(minutes=CHOPPY_MIN_SPACING_MINUTES)

        if not force_refresh_due and spacing_block:
            return cache["choppiness_score"]

    else:
        CHOPPINESS_SCORE_CACHE[symbol] = {
            "choppiness_score": 1.0,
            "last_refresh": datetime(2000, 1, 1),
        }
        cache = CHOPPINESS_SCORE_CACHE[symbol]

    # -----------------------------------------------------------------------
    # FETCH 1-MINUTE OHLCV
    # -----------------------------------------------------------------------
    try:
        df = exchange.fetch_1m_ohlcv(symbol)
    except Exception:
        CHOPPINESS_SCORE_CACHE[symbol]["choppiness_score"] = 1.0
        CHOPPINESS_SCORE_CACHE[symbol]["last_refresh"] = now
        return 1.0

    if df is None or len(df) < lookback:
        CHOPPINESS_SCORE_CACHE[symbol]["choppiness_score"] = 1.0
        CHOPPINESS_SCORE_CACHE[symbol]["last_refresh"] = now
        return 1.0

    df = df.tail(lookback)
    closes = df["close"].astype(float).values

    # -----------------------------------------------------------------------
    # CHOPPINESS CALCULATION
    # -----------------------------------------------------------------------
    net_move = abs(closes[-1] - closes[0])
    total_move = abs(closes[1:] - closes[:-1]).sum()

    if total_move <= 0:
        choppiness = 1.0
    else:
        efficiency = net_move / total_move

        # Normalize efficiency into choppiness
        # efficiency >= min_efficiency → choppiness approaches 0
        # efficiency → 0 → choppiness approaches 1
        choppiness = 1.0 - min(1.0, efficiency / min_efficiency)

    choppiness = float(np.clip(choppiness, 0.0, 1.0))

    # -----------------------------------------------------------------------
    # STORE RESULT
    # -----------------------------------------------------------------------
    CHOPPINESS_SCORE_CACHE[symbol]["choppiness_score"] = choppiness
    CHOPPINESS_SCORE_CACHE[symbol]["last_refresh"] = now

    return choppiness

def classify_trend_or_range(symbol, lookback=50, threshold=0.0003):
    """
    Cached wrapper for trend/range classification.
    """
    now = time.time()

    if symbol in TRENDRANGE_CACHE:
        ts, result = TRENDRANGE_CACHE[symbol]
        if now - ts < state.TRENDRANGE_CACHE_TIMEOUT:
            return result
        else:
            del TRENDRANGE_CACHE[symbol]

    result = classify_trend_or_range_real(symbol, lookback=lookback, threshold=threshold)
    TRENDRANGE_CACHE[symbol] = (now, result)
    return result

def calculate_easy_trend10_with_rsi(symbol, lookback=50, rsi_period=14,
                                   rsi_low_cutoff=30, rsi_high_cutoff=70,
                                   window_size=5):
    """
    'Easy Trend 10' variant (improved from Trend 9):
    - Uses overlapping sliding windows with log-return slopes.
    - Adaptive lookback (ATR-based).
    - Exponential weighting on recent slope segments.
    - Soft RSI boost instead of hard cutoff.
    - 60% majority rule.
    - Uses first 80% candles for slope calculations (vs 75%).
    - More tolerant fib retrace / breakout filters.
    - Adds EMA confirmation and trend persistence boost.
    - Returns {'score': float, 'stop_loss': float} or 0.0
    """
    def _clamp(x, lo, hi):
        return max(lo, min(x, hi))

    try:
        alive_conf_raw = alive_chart_score(symbol)
        choppy_conf_raw = 1.0 - choppiness_score(symbol)

        alive_chart_score_conf = _clamp(0.5 + 0.5 * alive_conf_raw, 0.5, 1.0)
        choppiness_score_conf = _clamp(0.5 + 0.5 * choppy_conf_raw, 0.5, 1.0)
    except Exception:
        alive_chart_score_conf = 0.75
        choppiness_score_conf = 0.75

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    # --- Adaptive lookback based on volatility (ATR ratio)
    try:
        atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        avg_price = df['close'].iloc[-1]
        current_price = avg_price
        vol_ratio = atr / avg_price if avg_price != 0 else 0
        lookback = int(max(50, min(150, 100 * vol_ratio)))  # 50–150 range
    except Exception as e:
        debug(f"[WARN] ATR or lookback calculation failed for {symbol}: {e}")
        atr = 0.0
        current_price = df['close'].iloc[-1] if not df.empty else 0.0
        pass

    # --- Use ohlc4 values
    ohlc4 = ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0).astype(float)
    values = ohlc4.tail(lookback).values

    # --- Split 80% early / 20% late
    split_idx = int(len(values) * 0.8)
    early_values = values[:split_idx]
    late_values = values[split_idx:]

    if len(early_values) <= 9:
        log_returns = np.diff(np.log(early_values))
        slope_normalized = np.mean(log_returns)
        return float(slope_normalized)

    # --- Calculate slopes on early values
    segment_slopes = []
    for i in range(len(early_values) - window_size + 1):
        segment = early_values[i:i + window_size]
        log_returns = np.diff(np.log(segment))
        mean_log_ret = np.mean(log_returns)
        vol_adj_slope = mean_log_ret / (np.std(segment) + 1e-8)
        segment_slopes.append(vol_adj_slope)
        debug(
            f"[DEBUG EASY TREND10] {symbol} | Window {i+1}/{len(early_values)-window_size+1} | "
            f"MeanLogRet={mean_log_ret:.6f}, VolAdjSlope={vol_adj_slope:.6f}"
        )

    # --- Weighted average of slopes (recent > past)
    weights = np.linspace(0.2, 1.0, len(segment_slopes))
    slope_normalized = np.average(segment_slopes, weights=weights)
    # Save base score for confidence multipliers
    base_score = slope_normalized

    positive_count = sum(1 for s in segment_slopes if s > 0)
    negative_count = sum(1 for s in segment_slopes if s < 0)
    required_count = int(len(segment_slopes) * 0.6)  # 60% rule

    first_candle = early_values[0]
    last_candle = early_values[-1]

    if not (
        (positive_count >= required_count and last_candle > first_candle)
        or (negative_count >= required_count and last_candle < first_candle)
    ):
        return 0.0

    debug(
        f"[DEBUG EASY TREND10] {symbol} | AvgSlope={slope_normalized:.6f}, "
        f"First={first_candle:.4f}, Last={last_candle:.4f}, "
        f"PosCount={positive_count}, NegCount={negative_count}"
    )

    # RSI
    rsi_conf = 1.0
    try:
        rsi = ta.RSI(df['close'], timeperiod=rsi_period).iloc[-1]
        rsi_distance = abs(rsi - 50.0) / 25.0   # 0 → 2
        rsi_conf = 1.0 + _clamp(rsi_distance * 0.05, -0.10, 0.10)
    except Exception:
        pass

    # EMA
    ema_conf = 1.0
    try:
        ema_fast = ta.EMA(df['close'], timeperiod=21).iloc[-1]
        ema_slow = ta.EMA(df['close'], timeperiod=55).iloc[-1]

        if base_score > 0:
            ema_conf = 1.15 if ema_fast > ema_slow else 0.85
        else:
            ema_conf = 1.15 if ema_fast < ema_slow else 0.85
    except Exception:
        pass

    # Candle
    candle_conf = 1.0

    # --- Trend persistence check (last 10 bars)
    recent_returns = np.diff(np.log(values[-10:]))
    if slope_normalized > 0 and np.mean(recent_returns) > 0:
        slope_normalized *= 1.3
    elif slope_normalized < 0 and np.mean(recent_returns) < 0:
        slope_normalized *= 1.3

    # --- Determine fib retrace levels
    early_high = np.max(early_values)
    early_low = np.min(early_values)
    fib_retrace_long = early_high - (1 - state.FIB_LEVEL) * (early_high - early_low)
    fib_retrace_short = early_low + (1 - state.FIB_LEVEL) * (early_high - early_low)

    # === Common trailing parameters ===
    minimum_trailing_length = 0.0025 * current_price  # 0.25% safety floor
    trailing_length = max(1.2 * atr, minimum_trailing_length)

    advice_data = {}

    advice_data["minimum_trailing_length"] = minimum_trailing_length
    advice_data["trailing_length"] = trailing_length

    # === LONG/SHORT side trailing parameters ===
    long_trailing_trigger_price = current_price + (current_price - fib_retrace_long)  # ≈ +1R profit
    short_trailing_trigger_price = current_price - (fib_retrace_short - current_price)  # ≈ +1R profit

    MINIMUM_STOP_LOSS_PERCENT=0.30
    # --- Relaxed breakout / retrace conditions
    if slope_normalized > 0:

        # --- Previous candle confirmation (bullish)
        try:
            prev_open = df['open'].iloc[-2]
            prev_close = df['close'].iloc[-2]
            prev_high = df['high'].iloc[-2]
            prev_low = df['low'].iloc[-2]

            body = abs(prev_close - prev_open)
            total_range = max(prev_high - prev_low, 1e-8)
            body_ratio = body / total_range
            close_position = (prev_close - prev_low) / total_range  # 0 = low, 1 = high

            nice_bull = (prev_close > prev_open) and (body_ratio >= 0.6) and (close_position >= 0.75)
            # Candle quality confidence (bullish)
            candle_conf = _clamp(0.9 + 0.2 * body_ratio, 0.9, 1.1)

            if not nice_bull:
                debug(f"[{symbol}] Dismissed LONG: previous candle not strong bullish.")
                return 0.0
        except Exception as e:
            debug(f"[WARN] Previous candle validation failed for {symbol}: {e}")
            return 0.0

        fib_073 = early_low + (early_high - early_low) * 0.73
        fib_0768 = early_low + (early_high - early_low) * 0.768
        if not (fib_073 <= current_price <= fib_0768):
            debug(f"[{symbol}] Dismissed LONG: current price not within 0.73–0.768 fibo range")
            return 0.0
        if (current_price < fib_retrace_long):
            debug(f"[{symbol}] Dismissed LONG: fib_retrace_long would trigger stop loss immediately.")
            return 0.0
        if abs((fib_retrace_long - current_price) / current_price) < (MINIMUM_STOP_LOSS_PERCENT * 0.01):
            debug(f"[{symbol}] Dismissed LONG: fib_retrace_long too close to current price (<0.3%)")
            return 0.0
        if np.max(late_values) > early_high * 1.005:  # allow 0.5% breakout
            debug(f"[{symbol}] Discarded LONG: breakout above early high")
            return 0.0
        if np.min(late_values) < fib_retrace_long * 0.995:  # allow wiggle room
            debug(f"[{symbol}] Discarded LONG: retraced below Fib tolerance")
            return 0.0

        advice_data["trailing_trigger_price"] = long_trailing_trigger_price
        advice_data["stop_loss"] = fib_retrace_long
        advice_data["take_profit"] = early_low + (early_high - early_low) * 0.893

        stop_distance_pct = abs(current_price - advice_data["stop_loss"]) / current_price
        stop_conf = _clamp(1.0 + (stop_distance_pct - 0.005) * 10.0, 0.85, 1.10)

        final_score = (
            slope_normalized
            * rsi_conf
            * ema_conf
            * candle_conf
            * stop_conf
            * alive_chart_score_conf
            * choppiness_score_conf
        )

        return {"score": float(final_score), "advice": advice_data, "raw_slope": base_score}

    elif slope_normalized < 0:

        # --- Previous candle confirmation (bearish)
        try:
            prev_open = df['open'].iloc[-2]
            prev_close = df['close'].iloc[-2]
            prev_high = df['high'].iloc[-2]
            prev_low = df['low'].iloc[-2]

            body = abs(prev_close - prev_open)
            total_range = max(prev_high - prev_low, 1e-8)
            body_ratio = body / total_range
            close_position = (prev_close - prev_low) / total_range  # 0 = low, 1 = high

            nice_bear = (prev_close < prev_open) and (body_ratio >= 0.6) and (close_position <= 0.25)
            # Candle quality confidence (bearish)
            candle_conf = _clamp(0.9 + 0.2 * body_ratio, 0.9, 1.1)

            if not nice_bear:
                debug(f"[{symbol}] Dismissed SHORT: previous candle not strong bearish.")
                return 0.0
        except Exception as e:
            debug(f"[WARN] Previous candle validation failed for {symbol}: {e}")
            return 0.0

        fib_073 = early_high - (early_high - early_low) * 0.73
        fib_0768 = early_high - (early_high - early_low) * 0.768
        if not (fib_0768 <= current_price <= fib_073):
            debug(f"[{symbol}] Dismissed SHORT: current price not within 0.73–0.768 inverse fibo range")
            return 0.0
        if (current_price > fib_retrace_short):
            debug(f"[{symbol}] Dismissed SHORT: fib_retrace_short would trigger stop loss immediately.")
            return 0.0
        if abs((fib_retrace_short - current_price) / current_price) < (MINIMUM_STOP_LOSS_PERCENT * 0.01):
            debug(f"[{symbol}] Dismissed SHORT: fib_retrace_short too close to current price (<0.3%)")
            return 0.0
        if np.min(late_values) < early_low * 0.995:  # allow 0.5% breakout
            debug(f"[{symbol}] Discarded SHORT: breakout below early low")
            return 0.0
        if np.max(late_values) > fib_retrace_short * 1.005:
            debug(f"[{symbol}] Discarded SHORT: retraced above Fib tolerance")
            return 0.0

        advice_data["trailing_trigger_price"] = short_trailing_trigger_price
        advice_data["stop_loss"] = fib_retrace_short
        advice_data["take_profit"] = early_high - (early_high - early_low) * 0.893

        stop_distance_pct = abs(current_price - advice_data["stop_loss"]) / current_price
        stop_conf = _clamp(1.0 + (stop_distance_pct - 0.005) * 10.0, 0.85, 1.10)

        final_score = (
            slope_normalized
            * rsi_conf
            * ema_conf
            * candle_conf
            * stop_conf
            * alive_chart_score_conf
            * choppiness_score_conf
        )

        return {"score": float(final_score), "advice": advice_data, "raw_slope": base_score}

    return float(slope_normalized)

def calculate_easy_trend9_with_rsi(symbol, lookback=50, rsi_period=14,
                                   rsi_low_cutoff=30, rsi_high_cutoff=70,
                                   window_size=5):
    """
    'Easy Trend 9' variant (improved from Trend 8):
    - Uses overlapping sliding windows with log-return slopes.
    - Adaptive lookback (ATR-based).
    - Exponential weighting on recent slope segments.
    - Soft RSI boost instead of hard cutoff.
    - 60% majority rule.
    - Uses first 80% candles for slope calculations (vs 75%).
    - More tolerant fib retrace / breakout filters.
    - Adds EMA confirmation and trend persistence boost.
    - Returns {'score': float, 'stop_loss': float} or 0.0
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    # --- Adaptive lookback based on volatility (ATR ratio)
    try:
        atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        avg_price = df['close'].iloc[-1]
        current_price = avg_price
        vol_ratio = atr / avg_price if avg_price != 0 else 0
        lookback = int(max(50, min(150, 100 * vol_ratio)))  # 50–150 range
    except Exception as e:
        debug(f"[WARN] ATR or lookback calculation failed for {symbol}: {e}")
        atr = 0.0
        current_price = df['close'].iloc[-1] if not df.empty else 0.0
        pass

    # --- Use ohlc4 values
    ohlc4 = ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0).astype(float)
    values = ohlc4.tail(lookback).values

    # --- Split 80% early / 20% late
    split_idx = int(len(values) * 0.8)
    early_values = values[:split_idx]
    late_values = values[split_idx:]

    if len(early_values) <= 9:
        log_returns = np.diff(np.log(early_values))
        slope_normalized = np.mean(log_returns)
        return float(slope_normalized)

    # --- Calculate slopes on early values
    segment_slopes = []
    for i in range(len(early_values) - window_size + 1):
        segment = early_values[i:i + window_size]
        log_returns = np.diff(np.log(segment))
        mean_log_ret = np.mean(log_returns)
        vol_adj_slope = mean_log_ret / (np.std(segment) + 1e-8)
        segment_slopes.append(vol_adj_slope)
        debug(
            f"[DEBUG EASY TREND9] {symbol} | Window {i+1}/{len(early_values)-window_size+1} | "
            f"MeanLogRet={mean_log_ret:.6f}, VolAdjSlope={vol_adj_slope:.6f}"
        )

    # --- Weighted average of slopes (recent > past)
    weights = np.linspace(0.2, 1.0, len(segment_slopes))
    slope_normalized = np.average(segment_slopes, weights=weights)

    positive_count = sum(1 for s in segment_slopes if s > 0)
    negative_count = sum(1 for s in segment_slopes if s < 0)
    required_count = int(len(segment_slopes) * 0.6)  # 60% rule

    first_candle = early_values[0]
    last_candle = early_values[-1]

    if not (
        (positive_count >= required_count and last_candle > first_candle)
        or (negative_count >= required_count and last_candle < first_candle)
    ):
        return 0.0

    debug(
        f"[DEBUG EASY TREND9] {symbol} | AvgSlope={slope_normalized:.6f}, "
        f"First={first_candle:.4f}, Last={last_candle:.4f}, "
        f"PosCount={positive_count}, NegCount={negative_count}"
    )

    # --- RSI soft boost (not filter)
    try:
        rsi = ta.RSI(df['close'], timeperiod=rsi_period).iloc[-1]
        rsi_boost = 1.0
        if 55 < rsi < 70:
            rsi_boost = 1.2  # bullish support
        elif 30 < rsi < 45:
            rsi_boost = 1.2  # bearish support
        slope_normalized *= rsi_boost
    except Exception:
        pass

    # --- EMA confirmation filter
    try:
        ema_fast = ta.EMA(df['close'], timeperiod=21).iloc[-1]
        ema_slow = ta.EMA(df['close'], timeperiod=55).iloc[-1]
        if ema_fast > ema_slow and slope_normalized > 0:
            slope_normalized *= 1.2
        elif ema_fast < ema_slow and slope_normalized < 0:
            slope_normalized *= 1.2
        else:
            slope_normalized *= 0.5
    except Exception:
        pass

    # --- Trend persistence check (last 10 bars)
    recent_returns = np.diff(np.log(values[-10:]))
    if slope_normalized > 0 and np.mean(recent_returns) > 0:
        slope_normalized *= 1.3
    elif slope_normalized < 0 and np.mean(recent_returns) < 0:
        slope_normalized *= 1.3

    # --- Determine fib retrace levels
    early_high = np.max(early_values)
    early_low = np.min(early_values)
    fib_retrace_long = early_high - (1 - state.FIB_LEVEL) * (early_high - early_low)
    fib_retrace_short = early_low + (1 - state.FIB_LEVEL) * (early_high - early_low)

    # === Common trailing parameters ===
    minimum_trailing_length = 0.0025 * current_price  # 0.25% safety floor
    trailing_length = max(1.2 * atr, minimum_trailing_length)

    advice_data = {}

    advice_data["minimum_trailing_length"] = minimum_trailing_length
    advice_data["trailing_length"] = trailing_length

    # === LONG/SHORT side trailing parameters ===
    long_trailing_trigger_price = current_price + (current_price - fib_retrace_long)  # ≈ +1R profit
    short_trailing_trigger_price = current_price - (fib_retrace_short - current_price)  # ≈ +1R profit

    MINIMUM_STOP_LOSS_PERCENT=0.30
    # --- Relaxed breakout / retrace conditions
    if slope_normalized > 0:
        if (current_price < fib_retrace_long):
            debug(f"[{symbol}] Dismissed LONG: fib_retrace_long would trigger stop loss immediately.")
            return 0.0
        if abs((fib_retrace_long - current_price) / current_price) < (MINIMUM_STOP_LOSS_PERCENT * 0.01):
            debug(f"[{symbol}] Dismissed LONG: fib_retrace_long too close to current price (<0.3%)")
            return 0.0
        if np.max(late_values) > early_high * 1.005:  # allow 0.5% breakout
            debug(f"[{symbol}] Discarded LONG: breakout above early high")
            return 0.0
        if np.min(late_values) < fib_retrace_long * 0.995:  # allow wiggle room
            debug(f"[{symbol}] Discarded LONG: retraced below Fib tolerance")
            return 0.0

        advice_data["trailing_trigger_price"] = long_trailing_trigger_price
        advice_data["stop_loss"] = fib_retrace_long
        return {"score": float(slope_normalized), "advice": advice_data}

    elif slope_normalized < 0:
        if (current_price > fib_retrace_short):
            debug(f"[{symbol}] Dismissed SHORT: fib_retrace_short would trigger stop loss immediately.")
            return 0.0
        if abs((fib_retrace_short - current_price) / current_price) < (MINIMUM_STOP_LOSS_PERCENT * 0.01):
            debug(f"[{symbol}] Dismissed SHORT: fib_retrace_short too close to current price (<0.3%)")
            return 0.0
        if np.min(late_values) < early_low * 0.995:  # allow 0.5% breakout
            debug(f"[{symbol}] Discarded SHORT: breakout below early low")
            return 0.0
        if np.max(late_values) > fib_retrace_short * 1.005:
            debug(f"[{symbol}] Discarded SHORT: retraced above Fib tolerance")
            return 0.0

        advice_data["trailing_trigger_price"] = short_trailing_trigger_price
        advice_data["stop_loss"] = fib_retrace_short
        return {"score": float(slope_normalized), "advice": advice_data}

    return float(slope_normalized)

def calculate_easy_trend8_with_rsi(symbol, lookback=50, rsi_period=14,
                                   rsi_low_cutoff=30, rsi_high_cutoff=70,
                                   window_size=5):
    """
    'Easy Trend 8' variant:
    - Uses overlapping sliding windows with log-return slopes.
    - 60% majority rule instead of 80%.
    - Range and RSI filters disabled.
    - Only use the first 75% of candles for slope calculations.
    - If latest 25% candles break above the earlier range → discard (Long)
    - If latest 25% candles retrace more than fib_level fib → discard (Long)
    - Return recommended StopLoss
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    # Use ohlc4 values
    ohlc4 = ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0).astype(float)
    values = ohlc4.tail(lookback).values

    # --- Split into early (75%) and late (25%)
    split_idx = int(len(values) * 0.75)
    early_values = values[:split_idx]
    late_values = values[split_idx:]

    if len(early_values) <= 9:
        # Simple slope only
        log_returns = np.diff(np.log(early_values))
        slope_normalized = np.mean(log_returns)
        return (float(slope_normalized))
    else:
        # --- Calculate slopes on early values
        segment_slopes = []
        for i in range(len(early_values) - window_size + 1):
            segment = early_values[i:i + window_size]

            log_returns = np.diff(np.log(segment))
            mean_log_ret = np.mean(log_returns)
            vol_adj_slope = mean_log_ret / (np.std(segment) + 1e-8)

            segment_slopes.append(vol_adj_slope)
            debug(
                f"[DEBUG EASY TREND8] {symbol} | Window {i+1}/{len(early_values)-window_size+1} | "
                f"MeanLogRet={mean_log_ret:.6f}, VolAdjSlope={vol_adj_slope:.6f}"
            )

        positive_count = sum(1 for s in segment_slopes if s > 0)
        negative_count = sum(1 for s in segment_slopes if s < 0)
        required_count = int(len(segment_slopes) * 0.6)  # ✅ 60% rule

        first_candle = early_values[0]
        last_candle = early_values[-1]

        if positive_count >= required_count and last_candle > first_candle:
            slope_normalized = np.mean(segment_slopes)
        elif negative_count >= required_count and last_candle < first_candle:
            slope_normalized = np.mean(segment_slopes)
        else:
            return 0.0

        print_with_date(
            f"[DEBUG EASY TREND8] {symbol} | (TOTAL) AvgSlope: {slope_normalized:.6f}, "
            f"First={first_candle:.4f}, Last={last_candle:.4f}, "
            f"PosCount={positive_count}, NegCount={negative_count}"
        )

        # --- Now decide trade direction
        early_high = np.max(early_values)
        early_low = np.min(early_values)

        fib_retrace_long  = early_high - (1 - state.FIB_LEVEL) * (early_high - early_low)
        fib_retrace_short = early_low  + (1 - state.FIB_LEVEL) * (early_high - early_low)

        if slope_normalized > 0:
            # Long trade → check upside breakout + downside retracement
            if np.max(late_values) > early_high:
                print_with_date(f"[{symbol}] Discarded LONG: breakout above early high ({np.max(late_values):.4f} > {early_high:.4f})")
                return 0.0
            if np.min(late_values) < fib_retrace_long:
                print_with_date(f"[{symbol}] Discarded LONG: retraced below {state.FIB_LEVEL:.3f} Fib")
                return 0.0
            return {"score": float(slope_normalized), "stop_loss": fib_retrace_long}

        elif slope_normalized < 0:
            # Short trade → check downside breakout + upside retracement
            if np.min(late_values) < early_low:
                print_with_date(f"[{symbol}] Discarded SHORT: breakout below early low ({np.min(late_values):.4f} < {early_low:.4f})")
                return 0.0
            if np.max(late_values) > fib_retrace_short:
                print_with_date(f"[{symbol}] Discarded SHORT: retraced above {state.FIB_LEVEL:.3f} Fib")
                return 0.0
            return {"score": float(slope_normalized), "stop_loss": fib_retrace_short}

    # ✅ Range filter disabled
    if False:
        max_price = np.max(values)
        min_price = np.min(values)
        range_pct = (max_price - min_price) / np.mean(values) * 100
        if range_pct < 0.5:
            return 0.0

    # ✅ RSI filter disabled
    if False:
        delta = np.diff(values)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-rsi_period:])
        avg_loss = np.mean(loss[-rsi_period:])
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi = 100 - (100 / (1 + rs))

        if rsi < rsi_low_cutoff or rsi > rsi_high_cutoff:
            return 0.0

    return (float(slope_normalized))

def calculate_easy_trend7_with_rsi(symbol, lookback=50, rsi_period=14,
                                   rsi_low_cutoff=30, rsi_high_cutoff=70,
                                   window_size=5):
    """
    'Easy Trend 7' variant:
    - Uses overlapping sliding windows with log-return slopes.
    - 60% majority rule instead of 80%.
    - Range and RSI filters disabled.
    - Only use the first 75% of candles for slope calculations.
    - If latest 25% candles break above the earlier range → discard (Long)
    - If latest 25% candles retrace more than 0.618 fib → discard (Long)
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    # Use ohlc4 values
    ohlc4 = ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0).astype(float)
    values = ohlc4.tail(lookback).values

    # --- Split into early (75%) and late (25%)
    split_idx = int(len(values) * 0.75)
    early_values = values[:split_idx]
    late_values = values[split_idx:]

    if len(early_values) <= 9:
        # Simple slope only
        log_returns = np.diff(np.log(early_values))
        slope_normalized = np.mean(log_returns)
    else:
        # --- Calculate slopes on early values
        segment_slopes = []
        for i in range(len(early_values) - window_size + 1):
            segment = early_values[i:i + window_size]

            log_returns = np.diff(np.log(segment))
            mean_log_ret = np.mean(log_returns)
            vol_adj_slope = mean_log_ret / (np.std(segment) + 1e-8)

            segment_slopes.append(vol_adj_slope)
            debug(
                f"[DEBUG EASY TREND7] {symbol} | Window {i+1}/{len(early_values)-window_size+1} | "
                f"MeanLogRet={mean_log_ret:.6f}, VolAdjSlope={vol_adj_slope:.6f}"
            )

        positive_count = sum(1 for s in segment_slopes if s > 0)
        negative_count = sum(1 for s in segment_slopes if s < 0)
        required_count = int(len(segment_slopes) * 0.6)  # ✅ 60% rule

        first_candle = early_values[0]
        last_candle = early_values[-1]

        if positive_count >= required_count and last_candle > first_candle:
            slope_normalized = np.mean(segment_slopes)
        elif negative_count >= required_count and last_candle < first_candle:
            slope_normalized = np.mean(segment_slopes)
        else:
            return 0.0

        print_with_date(
            f"[DEBUG EASY TREND7] {symbol} | (TOTAL) AvgSlope: {slope_normalized:.6f}, "
            f"First={first_candle:.4f}, Last={last_candle:.4f}, "
            f"PosCount={positive_count}, NegCount={negative_count}"
        )

        # --- Now decide trade direction
        early_high = np.max(early_values)
        early_low = np.min(early_values)

        fib_retrace_long  = early_high - (1 - 0.618) * (early_high - early_low)
        fib_retrace_short = early_low  + (1 - 0.618) * (early_high - early_low)

        if slope_normalized > 0:
            # Long trade → check upside breakout + downside retracement
            if np.max(late_values) > early_high:
                print_with_date(f"[{symbol}] Discarded LONG: breakout above early high ({np.max(late_values):.4f} > {early_high:.4f})")
                return 0.0
            if np.min(late_values) < fib_retrace_long:
                print_with_date(f"[{symbol}] Discarded LONG: retraced below 0.618 Fib")
                return 0.0

        elif slope_normalized < 0:
            # Short trade → check downside breakout + upside retracement
            if np.min(late_values) < early_low:
                print_with_date(f"[{symbol}] Discarded SHORT: breakout below early low ({np.min(late_values):.4f} < {early_low:.4f})")
                return 0.0
            if np.max(late_values) > fib_retrace_short:
                print_with_date(f"[{symbol}] Discarded SHORT: retraced above 0.618 Fib")
                return 0.0

    # ✅ Range filter disabled
    if False:
        max_price = np.max(values)
        min_price = np.min(values)
        range_pct = (max_price - min_price) / np.mean(values) * 100
        if range_pct < 0.5:
            return 0.0

    # ✅ RSI filter disabled
    if False:
        delta = np.diff(values)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-rsi_period:])
        avg_loss = np.mean(loss[-rsi_period:])
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi = 100 - (100 / (1 + rs))

        if rsi < rsi_low_cutoff or rsi > rsi_high_cutoff:
            return 0.0

    return float(slope_normalized)

def calculate_easy_trend6_with_rsi(symbol, lookback=50, rsi_period=14,
                                   rsi_low_cutoff=30, rsi_high_cutoff=70,
                                   window_size=5):
    """
    'Easy Trend 6' variant:
    - Uses overlapping sliding windows with log-return slopes.
    - 60% majority rule instead of 80%.
    - Range and RSI filters disabled.
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    # Use ohlc4 values
    ohlc4 = ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0).astype(float)
    values = ohlc4.tail(lookback).values

    if len(values) <= 9:
        log_returns = np.diff(np.log(values))
        slope_normalized = np.mean(log_returns)
    else:
        segment_slopes = []

        for i in range(len(values) - window_size + 1):
            segment = values[i:i + window_size]

            log_returns = np.diff(np.log(segment))
            mean_log_ret = np.mean(log_returns)
            vol_adj_slope = mean_log_ret / (np.std(segment) + 1e-8)

            segment_slopes.append(vol_adj_slope)
            debug(
                f"[DEBUG EASY TREND6] {symbol} | Window {i+1}/{len(values)-window_size+1} | "
                f"MeanLogRet={mean_log_ret:.6f}, VolAdjSlope={vol_adj_slope:.6f}"
            )

        positive_count = sum(1 for s in segment_slopes if s > 0)
        negative_count = sum(1 for s in segment_slopes if s < 0)
        required_count = int(len(segment_slopes) * 0.6)  # ✅ 60% rule

        first_candle = values[0]
        last_candle = values[-1]

        if positive_count >= required_count and last_candle > first_candle:
            slope_normalized = np.mean(segment_slopes)
        elif negative_count >= required_count and last_candle < first_candle:
            slope_normalized = np.mean(segment_slopes)
        else:
            return 0.0

        print_with_date(
            f"[DEBUG EASY TREND6] {symbol} | (TOTAL) AvgSlope: {slope_normalized:.6f}, "
            f"First={first_candle:.4f}, Last={last_candle:.4f}, "
            f"PosCount={positive_count}, NegCount={negative_count}"
        )

    # ✅ Range filter disabled
    if False:
        max_price = np.max(values)
        min_price = np.min(values)
        range_pct = (max_price - min_price) / np.mean(values) * 100
        if range_pct < 0.5:
            return 0.0

    # ✅ RSI filter disabled
    if False:
        delta = np.diff(values)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-rsi_period:])
        avg_loss = np.mean(loss[-rsi_period:])
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi = 100 - (100 / (1 + rs))

        if rsi < rsi_low_cutoff or rsi > rsi_high_cutoff:
            return 0.0

    return float(slope_normalized)

def calculate_easy_trend5_with_rsi(symbol, lookback=50, rsi_period=14,
                                   rsi_low_cutoff=30, rsi_high_cutoff=70,
                                   window_size=5):
    """
    'Easy Trend 5' using overlapping sliding windows with log-return slopes.
    Uses ohlc4 and weights consistency of slopes to determine trend.
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    # Use ohlc4 values
    ohlc4 = ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0).astype(float)
    values = ohlc4.tail(lookback).values

    if len(values) <= 9:
        # For short lookbacks: average log returns over whole period
        log_returns = np.diff(np.log(values))
        slope_normalized = np.mean(log_returns)
    else:
        segment_slopes = []

        # ✅ Overlapping sliding windows of size `window_size`
        for i in range(len(values) - window_size + 1):
            segment = values[i:i + window_size]

            # Log-return slope for this window
            log_returns = np.diff(np.log(segment))
            mean_log_ret = np.mean(log_returns)
            vol_adj_slope = mean_log_ret / (np.std(segment) + 1e-8)

            segment_slopes.append(vol_adj_slope)
            debug(
                f"[DEBUG EASY TREND5] {symbol} | Window {i+1}/{len(values)-window_size+1} | "
                f"MeanLogRet={mean_log_ret:.6f}, VolAdjSlope={vol_adj_slope:.6f}"
            )

        # ✅ Check consistency of slopes (80% majority rule)
        positive_count = sum(1 for s in segment_slopes if s > 0)
        negative_count = sum(1 for s in segment_slopes if s < 0)
        required_count = int(len(segment_slopes) * 0.8)

        first_candle = values[0]
        last_candle = values[-1]

        if positive_count >= required_count and last_candle > first_candle:
            slope_normalized = np.mean(segment_slopes)
        elif negative_count >= required_count and last_candle < first_candle:
            slope_normalized = np.mean(segment_slopes)
        else:
            return 0.0

        print_with_date(
            f"[DEBUG EASY TREND5] {symbol} | (TOTAL) AvgSlope: {slope_normalized:.6f}, "
            f"First={first_candle:.4f}, Last={last_candle:.4f}, "
            f"PosCount={positive_count}, NegCount={negative_count}"
        )

    # Range filter
    max_price = np.max(values)
    min_price = np.min(values)
    range_pct = (max_price - min_price) / np.mean(values) * 100
    if range_pct < 0.5:
        return 0.0

    # RSI on ohlc4
    delta = np.diff(values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-rsi_period:])
    avg_loss = np.mean(loss[-rsi_period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    if rsi < rsi_low_cutoff or rsi > rsi_high_cutoff:
        return 0.0

    return float(slope_normalized)

def calculate_easy_trend4_with_rsi(symbol, lookback=50, rsi_period=14,
                                   rsi_low_cutoff=30, rsi_high_cutoff=70):
    """
    Calculate 'easy trend 4' score using RSI filter.
    Uses ohlc4 and log-return based slopes per segment with volatility normalization.
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    # Use ohlc4 as smoothed price input
    ohlc4 = ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0).astype(float)
    values = ohlc4.tail(lookback).values

    if len(values) <= 9:
        # For very short lookback, use mean log return as overall slope
        log_returns = np.diff(np.log(values))
        slope_normalized = np.mean(log_returns)
    else:
        segment_size = 5
        num_segments = len(values) // segment_size
        values = values[-num_segments * segment_size:]

        segment_slopes = []
        for i in range(num_segments):
            segment = values[i * segment_size:(i + 1) * segment_size]

            # ✅ Use log returns for slope
            log_returns = np.diff(np.log(segment))
            mean_log_ret = np.mean(log_returns)

            # Volatility adjustment: divide by stdev of segment prices
            vol_adj_slope = mean_log_ret / (np.std(segment) + 1e-8)

            debug(
                f"[DEBUG EASY TREND4] {symbol} | Segment {i+1}/{num_segments} | "
                f"MeanLogRet={mean_log_ret:.6f}, VolAdjSlope={vol_adj_slope:.6f}"
            )

            segment_slopes.append(vol_adj_slope)

        positive_count = sum(1 for s in segment_slopes if s > 0)
        negative_count = sum(1 for s in segment_slopes if s < 0)
        required_count = int(len(segment_slopes) * 0.8)

        first_candle = values[0]
        last_candle = values[-1]

        if positive_count >= required_count and last_candle > first_candle:
            slope_normalized = sum(segment_slopes)
        elif negative_count >= required_count and last_candle < first_candle:
            slope_normalized = sum(segment_slopes)
        else:
            return 0.0

        print_with_date(
            f"[DEBUG EASY TREND4] {symbol} | (TOTAL) SlopeNormalized: {slope_normalized}, "
            f"First={first_candle:.4f}, Last={last_candle:.4f}"
        )

    # Range filter
    max_price = np.max(values)
    min_price = np.min(values)
    range_pct = (max_price - min_price) / np.mean(values) * 100
    if range_pct < 0.5:
        return 0.0

    # RSI using ohlc4
    delta = np.diff(values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-rsi_period:])
    avg_loss = np.mean(loss[-rsi_period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    if rsi < rsi_low_cutoff or rsi > rsi_high_cutoff:
        return 0.0

    return float(slope_normalized)

def calculate_easy_trend3_with_rsi(symbol, lookback=50, rsi_period=14,
                                   rsi_low_cutoff=30, rsi_high_cutoff=70):
    """
    Calculate an 'easy trend 3' score using RSI filter.
    Uses ohlc4 and requires overall price movement to match trend direction:
    - Uptrend: last candle > first candle
    - Downtrend: last candle < first candle
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    ohlc4 = ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0).astype(float)
    values = ohlc4.tail(lookback).values

    if len(values) <= 9:
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        slope_normalized = slope / np.mean(values)
    else:
        segment_size = 5
        num_segments = len(values) // segment_size
        values = values[-num_segments * segment_size:]

        segment_slopes = []
        for i in range(num_segments):
            segment = values[i * segment_size:(i + 1) * segment_size]
            start_price = segment[0]
            end_price = segment[-1]
            mean_price = np.mean(segment)
            raw_slope = end_price - start_price
            normalized_slope = raw_slope / mean_price

            debug(
                f"[DEBUG EASY TREND3] {symbol} | Segment {i+1}/{num_segments} | "
                f"Start={start_price:.4f}, End={end_price:.4f}, "
                f"RawSlope={raw_slope:.6f}, NormSlope={normalized_slope:.6f}"
            )

            segment_slopes.append(normalized_slope)

        positive_count = sum(1 for s in segment_slopes if s > 0)
        negative_count = sum(1 for s in segment_slopes if s < 0)
        required_count = int(len(segment_slopes) * 0.8)

        first_candle = values[0]
        last_candle = values[-1]

        # ✅ Positive logic: accept only when both the slope count AND price movement match
        if positive_count >= required_count and last_candle > first_candle:
            slope_normalized = sum(segment_slopes)
        elif negative_count >= required_count and last_candle < first_candle:
            slope_normalized = sum(segment_slopes)
        else:
            return 0.0

        print_with_date(
            f"[DEBUG EASY TREND3] {symbol} | (TOTAL) SlopeNormalized: {slope_normalized}, "
            f"First={first_candle:.4f}, Last={last_candle:.4f}"
        )

    max_price = np.max(values)
    min_price = np.min(values)
    range_pct = (max_price - min_price) / np.mean(values) * 100
    if range_pct < 0.5:
        return 0.0

    delta = np.diff(values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-rsi_period:])
    avg_loss = np.mean(loss[-rsi_period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    if rsi < rsi_low_cutoff or rsi > rsi_high_cutoff:
        return 0.0

    return float(slope_normalized)

def calculate_easy_trend2_with_rsi(symbol, lookback=50, rsi_period=14,
                                   rsi_low_cutoff=30, rsi_high_cutoff=70):
    """
    Calculate an 'easy trend 2' score using RSI filter.
    Uses ohlc4 (average of open, high, low, close) instead of close values.
    For >=10 candles, slope is based on start/end ohlc4 per segment and normalized.
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    # Compute ohlc4
    ohlc4 = ((df['open'] + df['high'] + df['low'] + df['close']) / 4.0).astype(float)
    values = ohlc4.tail(lookback).values

    # For <=9 candles, same as trendest but on ohlc4
    if len(values) <= 9:
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        slope_normalized = slope / np.mean(values)
    else:
        segment_size = 5
        num_segments = len(values) // segment_size
        values = values[-num_segments * segment_size:]  # trim to multiple of 5

        segment_slopes = []
        for i in range(num_segments):
            segment = values[i * segment_size:(i + 1) * segment_size]
            start_price = segment[0]
            end_price = segment[-1]
            mean_price = np.mean(segment)
            raw_slope = end_price - start_price
            normalized_slope = raw_slope / mean_price

            # Debug print for each segment
            debug(
                f"[DEBUG EASY TREND2] {symbol} | Segment {i+1}/{num_segments} | "
                f"Start={start_price:.4f}, End={end_price:.4f}, "
                f"RawSlope={raw_slope:.6f}, NormSlope={normalized_slope:.6f}"
            )

            segment_slopes.append(normalized_slope)

        # Require at least 80% of segment slopes to be positive or negative
        positive_count = sum(1 for s in segment_slopes if s > 0)
        negative_count = sum(1 for s in segment_slopes if s < 0)
        required_count = int(len(segment_slopes) * 0.8)

        if positive_count >= required_count:
            # Mostly uptrend
            pass
        elif negative_count >= required_count:
            # Mostly downtrend
            pass
        else:
            return 0.0

        slope_normalized = sum(segment_slopes)
        print_with_date(
            f"[DEBUG EASY TREND2] {symbol} | (TOTAL) SlopeNormalized: {slope_normalized}"
        )

    # Range filter: avoid range-bound symbols
    max_price = np.max(values)
    min_price = np.min(values)
    range_pct = (max_price - min_price) / np.mean(values) * 100
    if range_pct < 0.5:
        return 0.0

    # Compute RSI on ohlc4
    delta = np.diff(values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-rsi_period:])
    avg_loss = np.mean(loss[-rsi_period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    if rsi < rsi_low_cutoff or rsi > rsi_high_cutoff:
        return 0.0

    return float(slope_normalized)

def calculate_easy_trend_with_rsi(symbol, lookback=50, rsi_period=14,
                                  rsi_low_cutoff=30, rsi_high_cutoff=70):
    """
    Calculate an 'easy trend' score using RSI filter.
    For >=10 candles, slope is based on start/end closes per segment and normalized.
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    closes = df['close'].astype(float).tail(lookback).values

    # For <=9 candles, same as the trendest version
    if len(closes) <= 9:
        x = np.arange(len(closes))
        slope, _ = np.polyfit(x, closes, 1)
        slope_normalized = slope / np.mean(closes)
    else:
        # Divide into 5-candle segments
        segment_size = 5
        num_segments = len(closes) // segment_size
        closes = closes[-num_segments * segment_size:]  # trim to multiple of 5

        segment_slopes = []
        for i in range(num_segments):
            segment = closes[i * segment_size:(i + 1) * segment_size]
            start_price = segment[0]
            end_price = segment[-1]
            mean_price = np.mean(segment)
            raw_slope = end_price - start_price
            normalized_slope = raw_slope / mean_price

            # Debug print for each segment
            debug(
                f"[DEBUG EASY TREND] {symbol} | Segment {i+1}/{num_segments} | "
                f"Start={start_price:.4f}, End={end_price:.4f}, "
                f"RawSlope={raw_slope:.6f}, NormSlope={normalized_slope:.6f}"
            )

            segment_slopes.append(normalized_slope)

        # Require at least 80% of segment slopes to be positive or negative
        positive_count = sum(1 for s in segment_slopes if s > 0)
        negative_count = sum(1 for s in segment_slopes if s < 0)
        required_count = int(len(segment_slopes) * 0.8)

        if positive_count >= required_count:
            # Mostly uptrend
            pass
        elif negative_count >= required_count:
            # Mostly downtrend
            pass
        else:
            # Mixed trend, discard
            return 0.0

        slope_normalized = sum(segment_slopes)
        # Debug print for each segment
        print_with_date(
            f"[DEBUG EASY TREND] {symbol} | (TOTAL) SlopeNormalized: {slope_normalized}"
        )

    # Range filter: avoid range-bound symbols
    max_price = np.max(closes)
    min_price = np.min(closes)
    range_pct = (max_price - min_price) / np.mean(closes) * 100
    if range_pct < 0.5:
        return 0.0

    # Compute RSI
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-rsi_period:])
    avg_loss = np.mean(loss[-rsi_period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    # Apply RSI cutoffs
    if rsi < rsi_low_cutoff or rsi > rsi_high_cutoff:
        return 0.0

    return float(slope_normalized)

def calculate_trendest_with_rsi(symbol, lookback=50, rsi_period=14,
                                rsi_low_cutoff=30, rsi_high_cutoff=70):
    """
    Calculate a 'trendest' score with RSI filter.
    Requires all segments to have the same slope direction.
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    closes = df['close'].astype(float).tail(lookback).values

    # Handle short lookbacks as the old version
    if len(closes) <= 9:
        x = np.arange(len(closes))
        slope, _ = np.polyfit(x, closes, 1)
        slope_normalized = slope / np.mean(closes)
    else:
        # Divide into 5-candle segments
        segment_size = 5
        num_segments = len(closes) // segment_size
        closes = closes[-num_segments * segment_size:]  # trim to multiple of 5

        segment_slopes = []
        for i in range(num_segments):
            segment = closes[i * segment_size:(i + 1) * segment_size]
            x = np.arange(len(segment))
            seg_slope, _ = np.polyfit(x, segment, 1)
            segment_slopes.append(seg_slope / np.mean(segment))

        # Check if all slopes have the same sign
        all_positive = all(s > 0 for s in segment_slopes)
        all_negative = all(s < 0 for s in segment_slopes)
        if not (all_positive or all_negative):
            return 0.0

        # Slope is sum of segment slopes
        slope_normalized = sum(segment_slopes)

    # Range filter: avoid range-bound symbols
    max_price = np.max(closes)
    min_price = np.min(closes)
    range_pct = (max_price - min_price) / np.mean(closes) * 100
    if range_pct < 0.5:
        return 0.0

    # Compute RSI
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-rsi_period:])
    avg_loss = np.mean(loss[-rsi_period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    # Apply RSI cutoffs
    if rsi < rsi_low_cutoff or rsi > rsi_high_cutoff:
        return 0.0

    return float(slope_normalized)

def calculate_trend_with_rsi(symbol, lookback=50, rsi_period=14, rsi_low_percentile=10, rsi_high_percentile=90):
    """
    Calculate a trend score for a symbol based on slope and RSI filter.
    Returns a positive or negative value, or 0 for range-bound symbols.
    """

    # Fetch candles using the cached function
    df = exchange.fetch_5m_ohlcv(symbol)
    if (not isinstance(df, pd.DataFrame)):
        return 0.0
    if df is None or df.empty or len(df) < lookback:
        return 0.0

    closes = df['close'].astype(float).tail(lookback).values

    # 1️ - Compute slope using linear regression
    x = np.arange(len(closes))
    slope, _ = np.polyfit(x, closes, 1)

    # Normalize slope by price to make it relative
    slope_normalized = slope / np.mean(closes)

    # 2️ - Range filter: if max-min is small, consider it range-bound
    max_price = np.max(closes)
    min_price = np.min(closes)
    range_pct = (max_price - min_price) / np.mean(closes) * 100
    if range_pct < 0.5:  # threshold can be tuned
        return 0.0

    # 3️ - Compute RSI
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-rsi_period:])
    avg_loss = np.mean(loss[-rsi_period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    # 4️ - Apply RSI percentile filter
    if rsi < rsi_low_percentile or rsi > rsi_high_percentile:
        return 0.0

    return float(slope_normalized)

def calculate_ema_trend_score(symbol, lookback=50):
    df = exchange.fetch_5m_ohlcv(symbol)  # currently returns 5m candles
    if df is None or len(df) < lookback:
        return 0

    df['ema'] = df['close'].ewm(span=lookback, adjust=False).mean()
    # Slope = difference between last EMA and EMA N bars ago
    slope = df['ema'].iloc[-1] - df['ema'].iloc[-lookback]
    return slope

def calculate_atr(df, period=14, ma='SMA', ma_period=48):
    """
    Calculate the Average True Range (ATR) using specified moving average method.

    Args:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
        period (int): The period for True Range calculation (typically 14).
        ma (str): Type of moving average - 'SMA', 'EMA', 'RMA', or 'Highest'.
        ma_period (int): The period for the moving average (default is same as `period`).

    Returns:
        float: The latest ATR value.
    """
    if ma_period is None:
        ma_period = period

    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    ma = ma.upper()
    if ma == 'SMA':
        df['ATR'] = df['TR'].rolling(window=ma_period).mean()
    elif ma == 'EMA':
        df['ATR'] = df['TR'].ewm(span=ma_period, adjust=False).mean()
    elif ma == 'RMA':
        df['ATR'] = df['TR'].ewm(alpha=1 / ma_period, adjust=False).mean()
    elif ma == 'HIGHEST':
        df['ATR'] = df['TR'].rolling(window=ma_period).max()
    else:
        raise ValueError("Invalid ma type. Use 'SMA', 'EMA', 'RMA', or 'Highest'.")

    return df['ATR'].iloc[-1]

def calculate_trailing_start_from_atr(symbol, multiplier=2.125, ma='HIGHEST', ma_period=48):
    df = exchange.fetch_5m_ohlcv(symbol)
    if df is None:
        return None
    atr = calculate_atr(df, ma_period=ma_period, ma=ma)
    last_close = df['close'].iloc[-1]
    atr_percent = (atr / last_close) * 100
    trailing_start = round(atr_percent * multiplier, 2)
    debug(f"[ATR] {symbol} {ma}(ATR(ma_period)) = {atr:.2f}, % = {atr_percent:.2f}, TRAILING_START = {trailing_start}%")
    return trailing_start
