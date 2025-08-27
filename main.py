import asyncio
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ---- repo path hygiene -------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# prefer package folder over any shadowing config.py
if (ROOT / "config.py").exists() and (ROOT / "config").is_dir():
    sys.modules.pop("config", None)

# ---- imports after sys.path fix ----------------------------------------------
from utils.logger import setup_logger
from config.preset_loader import apply_preset
from config.config_manager import ConfigManager
from core.exchange_client import ExchangeClient
from core.bot_manager import BotManager


def parse_args():
    p = argparse.ArgumentParser(description="Ultimate Bybit Trading Bot (Master Ensemble)")
    p.add_argument("--preset", default="optimized_multi", help="JSON preset name or path (without .json ok)")
    p.add_argument("--trailing", default="normal", help="trailing-stop profile: tight|normal|wide")
    p.add_argument("--testnet", dest="testnet", action="store_true", help="force TESTNET on")
    p.add_argument("--live", dest="testnet", action="store_false", help="force LIVE on")
    p.set_defaults(testnet=None)
    return p.parse_args()


async def initialize(args) -> BotManager:
    """
    Build everything and return a ready BotManager.
    Precedence: .env -> JSON preset -> CLI flags (testnet)
    """
    # 1) Load .env first (baseline)
    load_dotenv()  # safe; doesn't overwrite existing env by default

    # 2) Apply JSON preset (overrides .env for whitelisted keys)
    apply_preset(args.preset)

    # If a trailing profile was provided on CLI, ignore any numeric overrides from the preset
    if args.trailing:
        os.environ.pop("TRAILING_PERCENT", None)
        os.environ.pop("TRAILING_BREAKEVEN_AT", None)
    
    # 3) CLI can still override testnet toggle explicitly
    if args.testnet is True:
        os.environ["TESTNET"] = "True"
    elif args.testnet is False:
        os.environ["TESTNET"] = "False"

    # 4) Logger
    logger = setup_logger()
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"Python version: {pyver}")
    testnet = os.getenv("TESTNET", "True").strip().lower() in ("1", "true", "yes", "y", "on")
    logger.info(f"Starting in {'TESTNET' if testnet else 'LIVE'} mode")

    # 5) Config -> Exchange -> Bot
    cfg = ConfigManager(preset=args.preset, trailing=args.trailing)
    ex = ExchangeClient(config=cfg)
    await ex.connect()                 # loads markets, sets sandbox if TESTNET, logs balance
    bot = BotManager(config=cfg)
    await bot.startup(exchange=ex)     # sets position mode, prints trade controls
    return bot


async def run_bot(args):
    bot = await initialize(args)
    try:
        await bot.start()  # enters trading loop
    except asyncio.CancelledError:
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    args = parse_args()

    # Banner
    print(
        "\n"
        "    ╔══════════════════════════════════════════╗\n"
        "    ║     ULTIMATE BYBIT TRADING BOT v2.0      ║\n"
        "    ║         Master Ensemble Edition          ║\n"
        "    ║                                          ║\n"
        f"    ║  Preset: {args.preset:<28}║\n"
        f"    ║  Trailing: {args.trailing:<26}║\n"
        "    ╚══════════════════════════════════════════╝\n"
    )

    # Windows & Py3.11 are fine; if needed, you can set a policy here.
    asyncio.run(run_bot(args))
