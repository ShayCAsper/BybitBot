# utils/logger.py
import os
import sys
import re
from pathlib import Path
from loguru import logger

# Regex to remove emoji / non-BMP glyphs (prevents � / ? boxes in classic consoles)
#_EMOJI_RE = re.compile(r"[\U00010000-\U0010FFFF]")
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0E\uFE0F\u200D]")

def _supports_ansi() -> bool:
    """
    True if the terminal likely supports ANSI.
    On Windows we will try to enable it with colorama first.
    """
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    # Windows Terminal / VS Code / ANSICON
    return (
        "WT_SESSION" in os.environ
        or os.environ.get("TERM_PROGRAM") == "vscode"
        or "ANSICON" in os.environ
    )


def setup_logger(level: str = "INFO", to_file: bool = True):
    """
    Configure Loguru:
      - Colorized console (PowerShell supported via colorama)
      - Emoji stripping on legacy consoles (toggle with NO_EMOJI=0/1)
      - Rotating file logs under ./logs/
    """
    logger.remove()

    # ---- Try to enable ANSI colors in classic Windows console
    colorize_flag = False
    if os.name == "nt":
        try:
            import colorama  # installed with click / black
            colorama.just_fix_windows_console()
            colorize_flag = True
        except Exception:
            colorize_flag = False

    # Fallback detection + manual override
    colorize_flag = (
        colorize_flag
        or _supports_ansi()
        or os.getenv("LOG_COLOR", "").lower() in ("1", "true", "yes", "y")
    )

    # ---- Emoji stripping (default on classic PowerShell)
    strip_emoji_default = (
        os.name == "nt"
        and "WT_SESSION" not in os.environ
        and os.environ.get("TERM_PROGRAM") != "vscode"
    )
    no_emoji_env = os.getenv("NO_EMOJI", "").lower() in ("1", "true", "yes", "y")
    strip_emoji = strip_emoji_default or no_emoji_env

    def _patch(record):
        if strip_emoji and record.get("message"):
            record["message"] = _EMOJI_RE.sub("", record["message"])
        return record

    # Older Loguru might not support configure(patcher=...)
    try:
        logger.configure(patcher=_patch)
    except TypeError:
        # If unsupported, skip patcher; you'll just see emojis if the console can show them
        pass

    console_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        level=level,
        colorize=colorize_flag,
        backtrace=False,
        diagnose=False,
        format=console_fmt,
        enqueue=True,
    )

    if to_file:
        Path("logs").mkdir(parents=True, exist_ok=True)

        file_fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"

        logger.add(
            "logs/bot.log",
            rotation="10 MB",
            retention="14 days",
            level="INFO",
            format=file_fmt,
            backtrace=False,
            diagnose=False,
            enqueue=True,
        )

        logger.add(
            "logs/errors.log",
            rotation="5 MB",
            retention="30 days",
            level="WARNING",
            format=file_fmt,
            backtrace=False,
            diagnose=False,
            enqueue=True,
        )

    return logger
