import requests
import json

def check_fidelity():
    symbol = "BTCUSDT"
    base_url = "https://api.binance.us/api/v3" # Using US endpoint for consistency
    
    print(f"--- Fidelity Check: {symbol} ---")
    
    # 1. Check AggTrades (Current Pipeline)
    print("\n[1] Checking GET /aggTrades (Flow/Eulerian)...")
    try:
        resp = requests.get(f"{base_url}/aggTrades", params={"symbol": symbol, "limit": 1})
        resp.raise_for_status()
        data = resp.json()
        if data:
            print("Sample AggTrade:")
            print(json.dumps(data[0], indent=2))
            # [id, price, qty, first_trade_id, last_trade_id, time, is_buyer_maker, is_best_match]
            print("Analysis: AggTrades contains 'first_trade_id' and 'last_trade_id' but NO 'order_id'.")
            print("Status: EULERIAN ONLY (Volume/Price Fields)")
    except Exception as e:
        print(f"Error fetching AggTrades: {e}")

    # 2. Check Raw Trades (Potential Lagrangian)
    print("\n[2] Checking GET /trades (Particle/Lagrangian?)...")
    try:
        # Note: /trades usually returns recent trades. 
        # API documentation: list of trades.
        resp = requests.get(f"{base_url}/trades", params={"symbol": symbol, "limit": 1})
        resp.raise_for_status()
        data = resp.json()
        if data:
            print("Sample Raw Trade:")
            print(json.dumps(data[0], indent=2))
            
            # Check for 'orderId', 'buyerOrderId', 'sellerOrderId'
            keys = data[0].keys()
            if "orderId" in keys or "buyerOrderId" in keys:
                print("Analysis: FOUND Order IDs! Particle tracking is feasible.")
                print("Status: LAGRANGIAN CAPABLE")
            else:
                print(f"Keys found: {list(keys)}")
                print("Analysis: No Order ID fields found in public REST response.")
                print("Status: EULERIAN ONLY")
    except Exception as e:
        print(f"Error fetching Trades: {e}")

if __name__ == "__main__":
    check_fidelity()
