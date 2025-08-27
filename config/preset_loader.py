# config/preset_loader.py
import os, json
from pathlib import Path

# Keys we allow a preset to set (maps 1:1 to your .env-driven ConfigManager)
ALLOWED_KEYS = {
    # strategies & symbols
    "ACTIVE_STRATEGIES", "TRADING_COINS",
    "ADVANCED_SCALPING_COINS", "SCALPING_COINS", "MOMENTUM_COINS",
    "MEAN_REVERSION_COINS", "PAIRS_COINS", "RSR_COINS",

    # session & cadence
    "SESSION_HOURS_UTC", "LOOP_LOG_EVERY", "LOOP_INTERVAL",
    "INTERVAL_ADV_SCALPING", "INTERVAL_SCALPING", "INTERVAL_MOMENTUM",
    "INTERVAL_MEANREV", "INTERVAL_ML", "INTERVAL_PAIRS", "INTERVAL_RSR",

    # trailing stop
    "TRAILING_ENABLED", "TRAILING_PERCENT", "TRAILING_BREAKEVEN_AT",

    # risk
    "DAILY_LOSS_CAP_PCT", "MAX_DAILY_TRADES", "MAX_POSITIONS",
    "MIN_RR_SCALP", "MIN_RR_MOMENTUM", "MIN_RR_MEANREV",
    "MAX_CONSEC_LOSSES", "LOSS_COOLDOWN_MIN",

    # microstructure / quality
    "MAX_SPREAD_BPS_SCALP", "MAX_SPREAD_BPS_DEFAULT", "MIN_DEPTH_USD",
    "MIN_LIQUIDITY_USD", "MIN_VOL_RATIO",

    # scalper tuning
    "ADV_SCALP_MIN_IMBALANCE", "ADV_SCALP_MA_GAP_BPS",

    # toggles
    "FORCE_ONE_WAY_MODE", "TESTNET",
}

def _to_env_value(v):
    # All envs are strings; keep booleans/numbers readable.
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)

def _preset_path(preset_name: str) -> Path:
    base = Path(__file__).parent / "presets"
    if preset_name.endswith(".json"):
        return Path(preset_name) if Path(preset_name).is_file() else base / preset_name
    return base / f"{preset_name}.json"

def apply_preset(preset_name: str) -> dict:
    """
    Load JSON preset and export its keys into os.environ (whitelisted by ALLOWED_KEYS).
    Precedence: CLI args > JSON preset > existing env > defaults in code.
    Returns the loaded dict (useful for logging).
    """
    path = _preset_path(preset_name)
    if not path.exists():
        # no hard fail; you may want to run with pure .env
        print(f"[preset_loader] Preset not found: {path}. Running with .env/defaults.")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Preset must be a JSON object: {path}")

    applied = {}
    for k, v in data.items():
        if k in ALLOWED_KEYS:
            os.environ[k] = _to_env_value(v)
            applied[k] = os.environ[k]
        else:
            # ignore unknown keys silently (keeps presets future-proof)
            pass

    print(f"[preset_loader] Applied preset: {path.name} ({len(applied)} keys)")
    return data
