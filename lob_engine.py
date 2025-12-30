import requests
import pandas as pd
import numpy as np
import time

class BinanceFetcher:
    def __init__(self, symbol="BTCUSDT"):
        self.base_url = "https://api.binance.com/api/v3"
        self.symbol = symbol

    def fetch_agg_trades(self, limit=1000):
        """
        Fetches compressed text/tick data (AggTrades).
        Represents the 'Flow' or 'Velocity' of the market particles.
        """
        urls = [
            f"https://api.binance.com/api/v3/aggTrades",
            f"https://api.binance.us/api/v3/aggTrades"
        ]
        
        for base_url in urls:
            params = {"symbol": self.symbol, "limit": limit}
            try:
                print(f"Trying {base_url}...")
                resp = requests.get(base_url, params=params, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data)
            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch from {base_url}: {e}")
                continue
        
        print("All APIs failed (Geo-blocking?). Generating synthetic L3 data for verification.")
        return self._generate_synthetic_trades(limit)

    def _parse_response(self, data):
        # [id, price, qty, first_trade_id, last_trade_id, time, is_buyer_maker, is_best_match]
        df = pd.DataFrame(data, columns=["a", "p", "q", "f", "l", "T", "m", "M"])
        df["p"] = df["p"].astype(float) # Price
        df["q"] = df["q"].astype(float) # Quantity
        df["T"] = pd.to_datetime(df["T"], unit="ms")
        return df

    def _generate_synthetic_trades(self, limit):
        # Generate random walk trades to simulate LOB flow
        base_price = 50000.0
        prices = [base_price]
        for _ in range(limit):
            change = np.random.normal(0, 5)
            prices.append(prices[-1] + change)
        
        df = pd.DataFrame({
            "a": range(limit + 1),
            "p": prices,
            "q": np.abs(np.random.normal(0.1, 0.05, limit + 1)),
            "f": range(limit + 1),
            "l": range(limit + 1),
            "T": pd.date_range(start="2024-01-01", periods=limit + 1, freq="100ms"),
            "m": np.random.choice([True, False], limit + 1), # Buyer maker?
            "M": True
        })
        return df

class LOBState:
    def __init__(self):
        pass

    def vectorize_flow(self, trades_df, window_seconds=1):
        """
        Converts tick data into Hydrodynamic State Vectors.
        State S = [Price, Volume_Flow, Trade_Imbalance, Volatility_Proxy]
        This corresponds to [Pressure, Velocity, External_Force, Temperature].
        """
        if trades_df.empty:
            return None

        # Resample to fixed grid (Hydrodynamic Grid)
        trades_df.set_index("T", inplace=True)
        
        # VWAP (Volume Weighted Average Price) ~ Local Pressure
        # vwap = lambda x: np.average(x["p"], weights=x["q"]) if sum(x["q"]) > 0 else np.mean(x["p"])
        
        ohlc = trades_df["p"].resample(f"{window_seconds}s").ohlc()
        volume = trades_df["q"].resample(f"{window_seconds}s").sum()
        
        # Net Flow (Buy Vol - Sell Vol)
        # is_buyer_maker=True -> Sell initiated (Taker is Seller)
        # is_buyer_maker=False -> Buy initiated (Taker is Buyer)
        buy_vol = trades_df[~trades_df["m"]]["q"].resample(f"{window_seconds}s").sum()
        sell_vol = trades_df[trades_df["m"]]["q"].resample(f"{window_seconds}s").sum()
        net_flow = buy_vol - sell_vol # ~ Velocity field
        
        # Construct State Matrix
        state = pd.DataFrame({
            "pressure": ohlc["close"], # P
            "temperature": ohlc["high"] - ohlc["low"], # Volatility/Temp
            "mass_flux": volume, # Total Mass moved
            "velocity": net_flow # Directional Flow
        }).dropna()
        
        return state

if __name__ == "__main__":
    print("--- Phase 4b: Data Upgrade (LOB) ---")
    fetcher = BinanceFetcher(symbol="BTCUSDT")
    print(f"Fetching tick data for {fetcher.symbol}...")
    
    trades = fetcher.fetch_agg_trades(limit=5000)
    print(f"Fetched {len(trades)} trades.")
    
    constructor = LOBState()
    hydro_state = constructor.vectorize_flow(trades)
    
    print("\n[Hydrodynamic State Sample]")
    if hydro_state is not None:
        print(hydro_state.head())
        # Save for Econophysics model
        hydro_state.to_csv("lob_hydro_state.csv")
        print("\nSaved state to lob_hydro_state.csv")
    else:
        print("No state generated.")
