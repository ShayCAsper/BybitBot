"""
Helper utilities shared by scalping strategies.
"""
from typing import Dict, Any, Tuple

def compute_spread_and_mid(orderbook: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Returns (spread, best_bid, best_ask). Spread is fractional (e.g., 0.0002 for 2 bps).
    """
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    if not bids or not asks:
        return 0.0, 0.0, 0.0
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    if best_bid <= 0:
        return 0.0, best_bid, best_ask
    spread = (best_ask - best_bid) / best_bid
    return spread, best_bid, best_ask

def orderbook_imbalance(orderbook: Dict[str, Any], depth: int = 5) -> float:
    """
    Simple depth imbalance: (bid_size - ask_size) / (bid_size + ask_size)
    """
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    db = sum(float(b[1]) for b in bids[:depth])
    da = sum(float(a[1]) for a in asks[:depth])
    denom = (db + da)
    if denom <= 0:
        return 0.0
    return (db - da) / denom

def nearest_wall_proximity(orderbook: Dict[str, Any], current_price: float, side: str, depth: int = 10, bps_thresh: float = 5.0) -> bool:
    """
    Check if there is a 'wall' (large size) near the price within bps_thresh basis points.
    Heuristic: top N levels, the max size level is considered a wall.
    """
    if current_price <= 0:
        return False
    arr = orderbook.get("bids") if side == "buy" else orderbook.get("asks")
    arr = arr or []
    if not arr:
        return False
    # choose the level with largest size among first N
    candidate = max(arr[:depth], key=lambda x: float(x[1]))
    wall_price = float(candidate[0])
    bps = abs(wall_price - current_price) / current_price * 1e4  # basis points
    return bps <= bps_thresh
