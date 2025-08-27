import asyncio
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

# ensure local imports work when running from repo root
import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# prefer package folder over any config.py shadow
if (ROOT / "config.py").exists() and (ROOT / "config").is_dir():
    sys.modules.pop("config", None)

from utils.logger import setup_logger
from config.config_manager import ConfigManager
from core.exchange_client import ExchangeClient
from core.bot_manager import BotManager


async def initialize(args) -> BotManager:
    """
    Build everything and return a ready BotManager.
    """
    # 1) env + logger
    load_dotenv()  # loads local .env; safe because .env is in .gitignore
    logger = setup_logger()

    # allow CLI to override TESTNET
    if args.testnet is True:
        os.environ["TESTNET"] = "True"
    elif args.testnet is False:
        os.environ["TESTNET"] = "False"

    testnet = os.getenv("TESTNET", "True").lower() in ("1", "true", "yes", "y")
    logger.info(f"Starting in {'TESTNET' if testnet else 'LIVE'} mode")

    # 2) config
    cfg = ConfigManager(preset=args.preset, trailing=args.trailing)

    # 3) exchange
    ex = ExchangeClient(config=cfg)
    await ex.connect()  # loads markets, sets sandbox if TESTNET, logs balance

    # 4) bot manager (owns strategies, risk, allocator)
    bot = BotManager(config=cfg)
    await bot.startup(exchange=ex)  # sets position mode (One-Way), prints trade controls

    return bot


async def run_bot(args):
    bot = await initialize(args)
    try:
        await bot.start()  # enters trading loop
    except asyncio.CancelledError:
        pass
    finally:
        await bot.stop()


def parse_args():
    p = argparse.ArgumentParser(description="Ultimate Bybit Trading Bot (Master Ensemble)")
    p.add_argument("--preset", default="optimized_multi", help="config preset name")
    p.add_argument("--trailing", default="normal", help="trailing-stop profile")
    p.add_argument("--testnet", dest="testnet", action="store_true", help="force TESTNET on")
    p.add_argument("--live", dest="testnet", action="store_false", help="force LIVE on")
    p.set_defaults(testnet=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("\n"
          "    ╔══════════════════════════════════════════╗\n"
          "    ║     ULTIMATE BYBIT TRADING BOT v2.0      ║\n"
          "    ║         Master Ensemble Edition          ║\n"
          "    ║                                          ║\n"
          f"    ║  Preset: {args.preset:<28}║\n"
          f"    ║  Trailing: {args.trailing:<26}║\n"
          "    ╚══════════════════════════════════════════╝\n")

    asyncio.run(run_bot(args))
