# simulated_btse.py
from flask import Flask, request, jsonify
import random
import time

app = Flask(__name__)

# Simulated in-memory order book and trades
db = {
    "positions": [],
    "orders": [],
    "trades": []
}

@app.route("/api/v2.2/order", methods=["POST"])
def place_order():
    data = request.get_json()
    position_id = str(random.randint(100000, 999999))
    order_id = str(random.randint(1000000, 9999999))
    price = round(random.uniform(25000, 30000), 2)
    
    db["positions"].append({
        "positionId": position_id,
        "symbol": data["symbol"],
        "size": data["size"],
        "side": data["side"]
    })

    db["orders"].append({
        "orderID": order_id,
        "positionId": position_id,
        "price": price
    })

    return jsonify([{"positionId": position_id, "orderID": order_id, "price": price}])

@app.route("/api/v2.2/price")
def get_price():
    return jsonify([{"lastPrice": round(random.uniform(25000, 30000), 2)}])

@app.route("/api/v2.2/user/positions")
def get_positions():
    return jsonify(db["positions"])

@app.route("/api/v2.2/user/trade_history")
def get_trade_history():
    order_id = request.args.get("orderID") or request.args.get("clOrderID")
    for order in db["orders"]:
        if order["orderID"] == order_id:
            trade = {
                "total": str(round(random.uniform(-0.002, 0.002), 8)),
                "price": order["price"]
            }
            return jsonify([trade])
    return jsonify([])

if __name__ == "__main__":
    app.run(port=5001, debug=True)
