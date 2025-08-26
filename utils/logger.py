"""
Advanced Logging System
"""

import sys
from loguru import logger
from pathlib import Path

def setup_logger():
    """Setup the logging system"""
    
    # Remove default logger
    logger.remove()
    
    # Console logger with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # File logger for all logs
    logger.add(
        "logs/bot.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="zip"
    )
    
    # Separate file for errors
    logger.add(
        "logs/errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="ERROR",
        rotation="50 MB",
        retention="60 days",
        backtrace=True,
        diagnose=True
    )
    
    # Trade logger
    logger.add(
        "logs/trades.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        filter=lambda record: "trade" in record["extra"],
        rotation="100 MB",
        retention="90 days"
    )
    
    return logger

def get_logger(name: str):
    """Get a logger instance"""
    return logger.bind(name=name)