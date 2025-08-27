import sys
from pathlib import Path
from loguru import logger

def setup_logger(level: str = "INFO"):
    """
    Console + rotating file logs under ./logs/.
    - Colorized levels (INFO white, WARNING yellow, ERROR red)
    - Backtraces and diagnostics for exceptions
    - Rotating files
    """
    # Remove default handler
    logger.remove()

    # Ensure logs dir exists
    Path("logs").mkdir(parents=True, exist_ok=True)

    fmt = (
        "<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | "
        "<level>{level: <8}</level> | "
        "{name}:{function} - "
        "<level>{message}</level>"
    )

    # Console
    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format=fmt,
        backtrace=True,
        diagnose=False,   # set True if you want extremely verbose exception introspection
        enqueue=True,
    )

    # Files
    logger.add(
        "logs/bot.log",
        rotation="10 MB",
        retention="14 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )
    logger.add(
        "logs/errors.log",
        rotation="5 MB",
        retention="30 days",
        level="WARNING",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    return logger
