import sys
from pathlib import Path
from loguru import logger

def setup_logger(level: str = "INFO"):
    """
    Console + rotating file logs under ./logs/.
    Call this once from main.py.
    """
    # Remove default handler
    logger.remove()

    # Ensure logs dir exists
    Path("logs").mkdir(parents=True, exist_ok=True)

    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"

    # Console
    logger.add(sys.stdout, level=level, colorize=True, format=fmt)

    # Files
    logger.add("logs/bot.log", rotation="10 MB", retention="14 days", level="INFO", format=fmt)
    logger.add("logs/errors.log", rotation="5 MB", retention="14 days", level="WARNING", format=fmt)

    return logger
