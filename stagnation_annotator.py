import os
import csv
import time
from datetime import datetime

# Import system state and utils
import state
from utils import print_with_date
from exchange import btse as exchange

# FIXED IMPORTS:
# fetch_top_symbols_by_volume is in symbols_setup.py
from symbols_setup import fetch_top_symbols_by_volume 
# stagnation_score is in trading/trend.py
from trading.trend import stagnation_score  

# Configuration
CSV_FILENAME = "symbol_stagnation_labels.csv"

def run_annotator():
    # Initialize basic state variables needed by the system
    state.DEBUG_MODE = False
    state.CLIENT_NAME = "STAGNATION-ANNOTATOR"
    
    print_with_date("[ANNOTATOR] Starting Stagnation Labeling Session...")
    print_with_date("[INFO] Ensure api_daemon.py is running in another buffer.")
    
    # 1. Fetch symbols using the utility function
    symbols = fetch_top_symbols_by_volume(limit=1000)
    if not symbols:
        print_with_date("[ERROR] Could not fetch symbols. Check Daemon/Network.")
        return

    # Check if CSV exists to handle headers
    file_exists = os.path.isfile(CSV_FILENAME)

    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "stagnation_score"])

        for symbol in symbols:
            try:
                # 2. Reuse your exact stagnation_score logic
                score = stagnation_score(symbol)
                
                print("\n" + "="*50)
                print(f"SYMBOL: {symbol}")
                print(f"System Stagnation Score: {score:.4f}")
                print(f"Reference: 0.0 (Dead) <---> 1.0 (Alive)")
                print("="*50)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                    symbol, 
                    round(score, 4)
                ])
                
            except Exception as e:
                print_with_date(f"[ERROR] Processing {symbol}: {e}")

    print_with_date(f"[FINISHED] Session closed. Data saved to {CSV_FILENAME}")

if __name__ == "__main__":
    run_annotator()
