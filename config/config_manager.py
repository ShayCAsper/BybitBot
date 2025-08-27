import os
from dataclasses import dataclass
from typing import Dict, Any, List

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return str(v).lower() in ("1","true","yes","y","on")

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(float(os.getenv(key, default)))
    except Exception:
        return default

@dataclass
class TrailingConfig:
    mode: str
    percent: float
    breakeven_at: float

class ConfigManager:
    """
    Central place for presets, trailing config, symbols, strategy activation,
    cadence/intervals, and risk controls.
    """
    def __init__(self, preset: str = "optimized_multi", trailing: str = "normal"):
        self.preset = preset
        self.trailing_name = trailing

        # Symbols (comma-separated) default: BTC,ETH,SOL
        syms = os.getenv("TRADING_COINS", "BTC,ETH,SOL,DOGE,MATIC")
        self.symbols = [f"{s.strip()}/USDT:USDT" for s in syms.split(",") if s.strip()]

        # Active strategies
        self.active_strategies = os.getenv(
            "ACTIVE_STRATEGIES",
            "advanced_scalping,scalping,momentum,mean_reversion,pairs,rsr"
        ).replace(" ", "").split(",")

        # Risk knobs
        self.daily_loss_cap_pct = _env_float("DAILY_LOSS_CAP_PCT", 1.5)
        self.max_daily_trades   = _env_int("MAX_DAILY_TRADES", 20)
        self.max_positions      = _env_int("MAX_POSITIONS", 3)
        self.min_rr_scalp       = _env_float("MIN_RR_SCALP", 1.05)
        self.min_rr_momentum    = _env_float("MIN_RR_MOMENTUM", 1.20)
        self.min_rr_meanrev     = _env_float("MIN_RR_MEANREV", 1.20)
        self.max_consec_losses  = _env_int("MAX_CONSEC_LOSSES", 3)
        self.loss_cooldown_min  = _env_int("LOSS_COOLDOWN_MIN", 45)
        self.session_hours_utc  = os.getenv("SESSION_HOURS_UTC", "12-20")

        # Cadence
        self.loop_log_every     = _env_int("LOOP_LOG_EVERY", 10)
        self.base_interval      = _env_int("LOOP_INTERVAL", 5)
        self.strategy_intervals = {
            "advanced_scalping": _env_int("INTERVAL_ADV_SCALPING", 3),
            "scalping":          _env_int("INTERVAL_SCALPING", 3),
            "momentum":          _env_int("INTERVAL_MOMENTUM", 12),
            "mean_reversion":    _env_int("INTERVAL_MEANREV", 45),
            "ml":                _env_int("INTERVAL_ML", 90),
            "pairs":             _env_int("INTERVAL_PAIRS", 20),
            "rsr":               _env_int("INTERVAL_RSR", 20),
        }

        # Regime filter defaults
        self.min_liquidity_usd  = _env_float("MIN_LIQUIDITY_USD", 5e5)
        self.min_vol_ratio      = _env_float("MIN_VOL_RATIO", 0.0004)

        # Scalper nudges
        self.adv_min_imbalance  = _env_float("ADV_SCALP_MIN_IMBALANCE", 0.10)
        self.adv_ma_gap_bps     = _env_float("ADV_SCALP_MA_GAP_BPS", 5.0)

        # Position mode
        self.force_one_way      = _env_bool("FORCE_ONE_WAY_MODE", True)

        # Testnet
        self.testnet            = _env_bool("TESTNET", True)

        # Trailing
        self.trailing = self._build_trailing(trailing)

    def _build_trailing(self, name: str) -> TrailingConfig:
        name = (name or "normal").lower()
        if name == "tight":
            return TrailingConfig(mode="percentage", percent=1.0, breakeven_at=0.3)
        if name == "wide":
            return TrailingConfig(mode="percentage", percent=3.0, breakeven_at=1.0)
        # normal
        return TrailingConfig(mode="percentage", percent=2.0, breakeven_at=0.5)

    # Convenience getters
    def symbols_list(self) -> List[str]:
        return self.symbols

    def strategies(self) -> List[str]:
        return [s for s in self.active_strategies if s]

    def cadence(self) -> Dict[str, int]:
        return self.strategy_intervals

    def risk(self) -> Dict[str, Any]:
        return {
            "daily_loss_cap_pct": self.daily_loss_cap_pct,
            "max_daily_trades": self.max_daily_trades,
            "max_positions": self.max_positions,
            "min_rr": {
                "scalping": self.min_rr_scalp,
                "momentum": self.min_rr_momentum,
                "mean_reversion": self.min_rr_meanrev,
            },
            "max_consec_losses": self.max_consec_losses,
            "loss_cooldown_min": self.loss_cooldown_min,
            "session_hours_utc": self.session_hours_utc,
        }
