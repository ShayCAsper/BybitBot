import os
from dataclasses import dataclass
from typing import Dict, Any, List


# --------- small helpers for env parsing ---------
def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

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


# --------- trailing-stop configuration ---------
@dataclass
class TrailingConfig:
    mode: str           # "percentage" (for now)
    percent: float      # trail distance in %
    breakeven_at: float # move SL to entry once profit >= this %
    enabled: bool = True


class ConfigManager:
    """
    Central configuration:
      • strategy activation
      • symbol universes (global + per-strategy)
      • cadence/intervals
      • risk caps & sessions
      • trailing-stop settings

    You can override almost everything via .env.
    """

    def __init__(self, preset: str = "optimized_multi", trailing: str = "normal"):
        self.preset = preset
        self.trailing_name = trailing

        # ---------- Symbols ----------
        # Global universe (used if a strategy-specific list is not provided)
        syms = os.getenv("TRADING_COINS", "BTC,ETH,SOL,DOGE,MATIC")
        self.symbols: List[str] = [
            f"{s.strip()}/USDT:USDT" for s in syms.split(",") if s.strip()
        ]

        # Which strategies are active (comma-separated)
        self.active_strategies: List[str] = os.getenv(
            "ACTIVE_STRATEGIES",
            "advanced_scalping,scalping,momentum,mean_reversion,pairs,rsr"
        ).replace(" ", "").split(",")

        # Per-strategy symbol allowlists (optional)
        def _symenv(key: str, default_list: List[str]) -> List[str]:
            raw = os.getenv(key, "").strip()
            if not raw:
                return default_list
            return [f"{s.strip()}/USDT:USDT" for s in raw.split(",") if s.strip()]

        self._symbols_per_strategy: Dict[str, List[str]] = {
            "advanced_scalping": _symenv("ADVANCED_SCALPING_COINS", self.symbols),
            "scalping":          _symenv("SCALPING_COINS",          self.symbols),
            "momentum":          _symenv("MOMENTUM_COINS",          self.symbols),
            "mean_reversion":    _symenv("MEAN_REVERSION_COINS",    self.symbols),
            "pairs":             _symenv("PAIRS_COINS",             ["BTC/USDT:USDT", "ETH/USDT:USDT"]),
            "rsr":               _symenv("RSR_COINS",               self.symbols),
        }

        # ---------- Risk / session ----------
        self.daily_loss_cap_pct: float = _env_float("DAILY_LOSS_CAP_PCT", 1.5)  # pause after this drawdown today
        self.max_daily_trades:   int   = _env_int("MAX_DAILY_TRADES", 20)
        self.max_positions:      int   = _env_int("MAX_POSITIONS", 2)
        self.min_rr_scalp:       float = _env_float("MIN_RR_SCALP", 1.02)
        self.min_rr_momentum:    float = _env_float("MIN_RR_MOMENTUM", 1.20)
        self.min_rr_meanrev:     float = _env_float("MIN_RR_MEANREV", 1.20)
        self.max_consec_losses:  int   = _env_int("MAX_CONSEC_LOSSES", 3)
        self.loss_cooldown_min:  int   = _env_int("LOSS_COOLDOWN_MIN", 45)
        self.session_hours_utc:  str   = os.getenv("SESSION_HOURS_UTC", "12-20")  # e.g. "0-24" while testing

        # ---------- Cadence ----------
        self.loop_log_every: int = _env_int("LOOP_LOG_EVERY", 10)  # log every N loops
        self.base_interval:  int = _env_int("LOOP_INTERVAL", 5)    # seconds

        self.strategy_intervals: Dict[str, int] = {
            "advanced_scalping": _env_int("INTERVAL_ADV_SCALPING", 3),
            "scalping":          _env_int("INTERVAL_SCALPING", 3),
            "momentum":          _env_int("INTERVAL_MOMENTUM", 12),
            "mean_reversion":    _env_int("INTERVAL_MEANREV", 45),
            "ml":                _env_int("INTERVAL_ML", 90),
            "pairs":             _env_int("INTERVAL_PAIRS", 20),
            "rsr":               _env_int("INTERVAL_RSR", 20),
        }

        # ---------- Regime / quality filters (used by some strategies) ----------
        self.min_liquidity_usd: float = _env_float("MIN_LIQUIDITY_USD", 5e5)
        self.min_vol_ratio:     float = _env_float("MIN_VOL_RATIO", 0.0004)

        # ---------- Advanced scalper nudges ----------
        self.adv_min_imbalance: float = _env_float("ADV_SCALP_MIN_IMBALANCE", 0.10)
        self.adv_ma_gap_bps:    float = _env_float("ADV_SCALP_MA_GAP_BPS", 5.0)

        # ---------- Position mode ----------
        self.force_one_way: bool = _env_bool("FORCE_ONE_WAY_MODE", True)

        # ---------- Testnet toggle ----------
        self.testnet: bool = _env_bool("TESTNET", True)

        # ---------- Trailing stop ----------
        self.trailing: TrailingConfig = self._build_trailing(self.trailing_name)

    # -- trailing profile builder (+ env overrides) --
    def _build_trailing(self, name: str) -> TrailingConfig:
        name = (name or "normal").lower()

        # presets
        if name == "tight":
            percent = 1.0
            breakev = 0.3
        elif name == "wide":
            percent = 3.0
            breakev = 1.0
        else:  # "normal"
            percent = 2.0
            breakev = 0.5

        # env overrides
        env_percent = _env_float("TRAILING_PERCENT", -1.0)
        env_break   = _env_float("TRAILING_BREAKEVEN_AT", -1.0)
        enabled     = _env_bool("TRAILING_ENABLED", True)

        if env_percent > 0: percent = env_percent
        if env_break   > 0: breakev = env_break

        return TrailingConfig(mode="percentage", percent=percent, breakeven_at=breakev, enabled=enabled)

    # ---------- public getters used by the bot ----------
    def symbols_list(self) -> List[str]:
        """Global fallback universe."""
        return self.symbols

    def strategy_symbols(self, name: str) -> List[str]:
        """Universe for a given strategy (falls back to global if not set)."""
        return self._symbols_per_strategy.get(name, self.symbols)

    def strategies(self) -> List[str]:
        """Active strategy names."""
        return [s for s in self.active_strategies if s]

    def cadence(self) -> Dict[str, int]:
        """Per-strategy loop intervals (seconds)."""
        return self.strategy_intervals

    def risk(self) -> Dict[str, Any]:
        """Structured risk/session settings consumed by BotManager."""
        return {
            "daily_loss_cap_pct": self.daily_loss_cap_pct,
            "max_daily_trades":   self.max_daily_trades,
            "max_positions":      self.max_positions,
            "min_rr": {
                "scalping":       self.min_rr_scalp,
                "momentum":       self.min_rr_momentum,
                "mean_reversion": self.min_rr_meanrev,
            },
            "max_consec_losses":  self.max_consec_losses,
            "loss_cooldown_min":  self.loss_cooldown_min,
            "session_hours_utc":  self.session_hours_utc,
        }
