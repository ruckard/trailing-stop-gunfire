# Trailing Stop Gunfire

## Introduction

This is trading script based on firing several trailing stop trades for a single symbol.
It keeps re-opening trailing stops that do end in a win trade.

## Setup

Copy `config.py.sample` as `config.py` and edit it to use your own BTSE futures keys.

You want all the symbols (The 20 symbols with most volume) to be setup like this:

- Isolated mode ( No cross mode )
- Position Mode: Multiple Mode ( No one-way mode )
- Leverage: 1x

You can use `python symbols_setup.py` for doing that in an automatic way.

## Usage

Just run:

```
python main.py
```

in a screen, tmux, byobu session.

## Output examples

### Resuming from save state

```
[2025-07-11 22:39:46] [ATR] BTC-PERP ATR(14) = 1677.16, % = 1.51, TRAILING_START = 0.51%
[2025-07-11 22:39:47] [ATR] ETH-PERP ATR(14) = 64.00, % = 2.35, TRAILING_START = 0.8%
[2025-07-11 22:39:48] [CONTRACTS_MAP] {'BTC-PERP': 1, 'ETH-PERP': 3}
[2025-07-11 22:39:48] [POSITIONS LOADED FROM DB]
[2025-07-11 22:39:48] [STORED] BTC-PERP | LONG | Callback: 0.64%
[2025-07-11 22:39:48] [STORED] BTC-PERP | SHORT | Callback: 0.64%
[2025-07-11 22:39:48] [STORED] BTC-PERP | LONG | Callback: 0.88%
[2025-07-11 22:39:48] [STORED] BTC-PERP | SHORT | Callback: 0.88%
[2025-07-11 22:39:48] [POSITIONS LOADED FROM DB]
[2025-07-11 22:39:48] [STORED] ETH-PERP | LONG | Callback: 0.87%
[2025-07-11 22:39:48] [STORED] ETH-PERP | SHORT | Callback: 0.87%
[2025-07-11 22:39:48] [STORED] ETH-PERP | LONG | Callback: 1.16%
[2025-07-11 22:39:48] [STORED] ETH-PERP | SHORT | Callback: 1.16%
[2025-07-11 22:39:48] CSCSCSCS
```

### Not reopening a lost trade

```
SC[2025-07-11 17:59:43] [ERROR] Position with position_id: ETH-PERP-USDT|6#11 not found.                                                                                                     
[2025-07-11 17:59:43] [CLOSED?] ETH-PERP long-1 position_id not found. Checking trade history...                                                                                             
[2025-07-11 17:59:46] [CLOSED/TRADE] ETH-PERP long-1 | Realized PnL: -0.00402755                                                                                                             
[2025-07-11 17:59:46] [LOSS] Not reopening ETH-PERP long-1
```

### Reopening a won trade

```
SCSCSCSCSCSCSCSCSC[2025-07-11 17:54:34] [ERROR] Position with position_id: BTC-PERP-USDT|4#267 not found.                                                                                    
[2025-07-11 17:54:34] [CLOSED?] BTC-PERP long-1 position_id not found. Checking trade history...                                                                                             
[2025-07-11 17:54:37] [CLOSED/TRADE] BTC-PERP long-1 | Realized PnL: 0.00851985                                                                                                              
[2025-07-11 17:54:37] [WIN] Reopening BTC-PERP long-1                                                                                                                                        
[2025-07-11 17:54:40] [NEW] BTC-PERP | LONG | Callback: 0.88% 
```
