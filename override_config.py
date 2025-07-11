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
        #"TRAILING_START": 0.4,
        "TRAILING_STEP_MULTIPLIER": 0.375, # Step is going to be 37.5% of calculated trailing start
        "TRAILING_COUNT": 2,
        "CONTRACTS": 0.00001 / 0.00001 # Currency_number / Contract_size
    },
    "ETH-PERP": {
        #"TRAILING_START": 0.6,
        "TRAILING_STEP_MULTIPLIER": 0.33, # Step is going to be 33.0% of calculated trailing start
        "TRAILING_COUNT": 2,
        "CONTRACTS": 0.0001 / 0.0001 # Currency_number / Contract_size
    }
}

API_DELAY_MS = 500          # Delay (in milliseconds) between API requests
