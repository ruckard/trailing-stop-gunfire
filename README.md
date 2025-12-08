# Trailing Stop Gunfire

## Introduction

This trading system executes Fibonacci-based retracement entries that follow the prevailing market trend. It identifies dominant directional momentum using high-conviction impulse candles, then waits for price to retrace into predefined Fibonacci levels before initiating a continuation trade. The strategy’s core objective is to exploit trend-aligned pullbacks that occur after strong expansion moves, using volatility-adjusted thresholds and structural filters to improve signal quality and execution accuracy.

The Trailing Stop Gunfire name comes from initial implementations that did trailing stops.

## Setup

### Configuration

Copy `config.py.sample` as `config.py` and edit it to use your own BTSE futures keys.
You can optionally copy `override_config.py.sample` as `override_config.py` and edit it to your needs.

### BTSE Subaccount setup

Subaccounts are useful for separating trading strategies, managing portfolios, or testing bots in isolated environments. That's way we encourage you to use them.

#### Log in to Your BTSE Account

Go to [btse.com](https://www.btse.com) and sign in with your main account credentials.

#### Open the **Subaccount Management** Page

* Click your profile icon in the top-right corner.
* Select **Sub-account** from the dropdown menu.

#### Create Sub-account

* Click **+ Create Sub-account** button.

#### Fill Out Subaccount Details

* **Subaccount Alias Needed**: Enter a unique name for your subaccount that describes your strategy (E.g.: *gunfire1* ). Click on *Next:Advanced Settings*.

#### Fill Out Advanced Settings

* Make sure to leave blank the **Email** for Enable Custom Login because you don't actually needed.
* **Transaction Permissions**: Choose **Futures Trading** only.
* **API Key Name**: *gunfire1key*
* **Key permission**: Trading, Read. **Very important. Otherwise the script won't work as expected.**
* **IP access restrictions**: Unless you have a fixed public IP which you control or own you'd better be off with **Unrestricted (Less Secure)**.
* Click on **Confirm** button.

#### Security Verification

* Enter the **Verification Code** that has been sent to your main account email.
* Enter the 2-Factor Authentication code for BTSE.

#### Sub-account Created summary

Something similar to:

```
You have successfully created a sub-account **GunFire1** with username **mainusernamesub00003**
Advanced Settings
- Futures Trading enabled
- API Key Name: gunfire1key
- API Key: 1500b0a00c3a37e14b088b09c29ef026b9f9cbf46524de97571c45f01d7f81be
- API Secret: adc5ec6cac98a44b7357c49723dfef4762e011f58e6d02666102ed59e9ff9d15
This API Secret will not be shown again. Please store it securely now.
```

will be shown to you.

Make sure to write down securely both the **API Key** and the **API Secret**.

Finally click the **Done** button.

#### Enable API Access (Optional but Recommended)

If you're using the subaccount for automated trading (e.g., bots), enable **API access**:

* Go to the **API Management** tab inside the subaccount settings.
* Click **Create API Key**.
* Set permissions: Trading and Read. **Very important. Otherwise the script won't work as expected.**
* Save the API key and secret securely.

#### Confirm and Create

* Review the information.
* Click **Create Subaccount**.
* You may be prompted to enter your 2FA code for confirmation.

### BTSE Charts Setup

You want all the symbols (The 20 symbols with most volume) to be setup like this:

- Isolated mode ( No cross mode )
- Position Mode: Multiple Mode ( No one-way mode )
- Leverage: 1x

You can use `python symbols_setup.py` for doing that in an automatic way.


## Usage

To operate the system, you must run the API daemon and the trading engine in separate terminal buffers.

### 1. Start the API daemon

Run the following command in a persistent session (screen, tmux, byobu, or similar):

```
python api_daemon.py
```

This daemon manages all real‑time exchange interactions and must remain active while the trading engine is running.

### 2. Start the trading engine

In a **separate** buffer or session, run:

```
python main.py
```

The trading engine will connect to the daemon, execute the trading cycle, and manage runtime logic.

## Output Examples

Actual outputs will differ from the examples below because the system no longer uses trailing stops and now relies on the API daemon for data flow. These samples illustrate the general structure only.

### Example: Loading state on startup

```
[2025-07-11 22:39:48] [STATE] Loaded saved cycle and symbol context
[2025-07-11 22:39:48] [SYMBOLS] Active: BTC-PERP, ETH-PERP
[2025-07-11 22:39:48] [POSITIONS] Restored 4 open positions from database
```

### Example: Not reopening a losing trade

```
[2025-07-11 17:59:46] [CLOSED/TRADE] ETH-PERP long-1 | Realized PnL: -0.00402755
[2025-07-11 17:59:46] [LOSS] Not reopening ETH-PERP long-1
```

### Example: Reopening a winning trade

```
[2025-07-11 17:54:37] [CLOSED/TRADE] BTC-PERP long-1 | Realized PnL: 0.00851985
[2025-07-11 17:54:37] [WIN] Reopening BTC-PERP long-1
```
