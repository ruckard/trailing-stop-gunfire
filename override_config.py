# Please MAKE SURE that all the symbols down below (BTC-PERP, ETH-PERP,...) are setup in:
#
# - Isolated mode ( No cross mode )
# - Position Mode: Multiple Mode ( No one-way mode )
# - Leverage: 1x
#
# before running the script.

# === Optional Overrides ===

SYMBOL_CONFIGS = {
    "BTC-PERP": {
        # "TRAILING_COUNT": 3, Optional: 3 trailing orders. Default: 2
        # "SYMBOL_MULTIPLIER": 2.0  # Optional: double exposure. Default: 1.0
        # "TRAILING_STEP_MULTIPLIER": 0.33, # Optional: Step is going to be 33.0% of calculated trailing start. Default: 0.375
    },
    "ETH-PERP": {
        # "TRAILING_COUNT": 3, Optional: 3 trailing orders. Default: 2
        # "SYMBOL_MULTIPLIER": 2.0  # Optional: double exposure. Default: 1.0
        # "TRAILING_STEP_MULTIPLIER": 0.33, # Optional: Step is going to be 33.0% of calculated trailing start. Default: 0.375
    }
}

API_DELAY_MS = 500          # Delay (in milliseconds) between API requests
