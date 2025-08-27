# tools/effective_config.py
import argparse
import os

# Ensure package imports work when run with -m
from config.preset_loader import apply_preset
from config.config_manager import ConfigManager

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="optimized_multi")
    p.add_argument("--trailing", default="normal")
    args = p.parse_args()

    # Apply JSON preset BEFORE building ConfigManager (same as main.py)
    apply_preset(args.preset)

    # If CLI trailing profile is supplied, let it win over any numeric overrides
    if args.trailing:
        os.environ.pop("TRAILING_PERCENT", None)
        os.environ.pop("TRAILING_BREAKEVEN_AT", None)

    cfg = ConfigManager(preset=args.preset, trailing=args.trailing)

    print("\n=== EFFECTIVE CONFIG ===")
    print(f"Preset name: {args.preset}")
    print(f"Active strategies: {', '.join(cfg.strategies())}")
    print(f"Global symbols: {', '.join(cfg.symbols_list())}")
    for s in cfg.strategies():
        print(f"  {s:18} -> {', '.join(cfg.strategy_symbols(s))}")

    print("\nCadence (sec):")
    for k, v in cfg.cadence().items():
        print(f"  {k:18} = {v}")

    r = cfg.risk()
    print("\nRisk/session:")
    print(f"  max_positions        = {r['max_positions']}")
    print(f"  max_daily_trades     = {r['max_daily_trades']}")
    print(f"  daily_loss_cap_pct   = {r['daily_loss_cap_pct']}")
    print(f"  RR floors            = {r['min_rr']}")
    print(f"  session_hours_utc    = {r['session_hours_utc']}")

    t = cfg.trailing
    print("\nTrailing:")
    print(f"  enabled={t.enabled}, mode={t.mode}, percent={t.percent}%, breakeven_at={t.breakeven_at}%\n")

if __name__ == "__main__":
    main()
