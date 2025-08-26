#!/usr/bin/env python3
"""
Ultimate Bybit Trading Bot - Main Entry Point with Trailing Stop Configuration
"""

import asyncio
import signal
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.bot_manager import BotManager
from utils.logger import setup_logger
from config.config_manager import ConfigManager

# Setup logger
logger = setup_logger()

class TradingBot:
    def __init__(self, preset=None, trailing_stop=None, custom_config=None):
        self.bot_manager = None
        self.running = False
        self.preset = preset
        self.trailing_stop = trailing_stop
        self.custom_config = custom_config or {}
        
    async def initialize(self):
        """Initialize all bot components"""
        try:
            logger.info("🚀 Initializing Ultimate Bybit Trading Bot...")
            
            # Load configuration
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            # Load preset if specified
            if self.preset:
                preset_path = Path(f"config/presets/{self.preset}.json")
                if preset_path.exists():
                    logger.info(f"📋 Loading preset: {self.preset}")
                    with open(preset_path, 'r') as f:
                        preset_config = json.load(f)
                        # Merge preset with main config
                        config['strategies'] = {**config['strategies'], **preset_config.get('strategies', {})}
                        config['risk'] = {**config['risk'], **preset_config.get('risk', {})}
                else:
                    logger.warning(f"⚠️ Preset '{self.preset}' not found, using default config")
            
            # Load trailing stop configuration
            if self.trailing_stop and self.trailing_stop != 'none':
                trailing_path = Path(f"config/trailing_stops/{self.trailing_stop}.json")
                if trailing_path.exists():
                    logger.info(f"📊 Loading trailing stop config: {self.trailing_stop}")
                    with open(trailing_path, 'r') as f:
                        trailing_config = json.load(f)
                        config['risk'] = {**config['risk'], **trailing_config}
                else:
                    logger.warning(f"⚠️ Trailing stop config '{self.trailing_stop}' not found")
            elif self.trailing_stop == 'none':
                config['risk']['trailing_stop'] = {'enabled': False}
            
            # Apply custom configuration overrides
            if self.custom_config:
                if 'trail_distance' in self.custom_config:
                    if 'trailing_stop' not in config['risk']:
                        config['risk']['trailing_stop'] = {}
                        config['risk']['trailing_stop']['distance'] = self.custom_config['trail_distance']
                        config['risk']['trailing_stop']['enabled'] = True
                
                if self.custom_config.get('breakeven'):
                    if 'trailing_stop' not in config['risk']:
                        config['risk']['trailing_stop'] = {}
                    config['risk']['trailing_stop']['breakeven_enabled'] = True
                    config['risk']['trailing_stop']['breakeven_trigger'] = 0.005
                
                if self.custom_config.get('partial_close'):
                    if 'trailing_stop' not in config['risk']:
                        config['risk']['trailing_stop'] = {}
                    config['risk']['trailing_stop']['partial_close_enabled'] = True
                
                if 'strategies' in self.custom_config:
                    config['strategies']['active'] = self.custom_config['strategies']
            
            # Log active configuration
            active_strategies = config['strategies'].get('active', [])
            logger.info(f"📊 Active strategies: {', '.join(active_strategies)}")
            
            # Log trailing stop configuration
            trailing_config = config['risk'].get('trailing_stop', {})
            if trailing_config.get('enabled', False):
                trailing_type = trailing_config.get('type', 'percentage')
                trailing_distance = trailing_config.get('distance', 0.02)
                logger.info(f"📈 Trailing Stop: {trailing_type} @ {trailing_distance:.1%}")
                if trailing_config.get('breakeven_enabled'):
                    logger.info(f"✅ Breakeven stop enabled @ {trailing_config.get('breakeven_trigger', 0.005):.1%}")
                if trailing_config.get('partial_close_enabled'):
                    logger.info("✅ Partial position closing enabled")
            else:
                logger.info("❌ Trailing stop disabled")
            
            # Initialize bot manager
            self.bot_manager = BotManager(config)
            await self.bot_manager.initialize()
            
            logger.info("✅ Bot initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize bot: {e}")
            return False
    
    async def start(self):
        """Start the trading bot"""
        if not await self.initialize():
            return
        
        self.running = True
        logger.info("📈 Starting trading bot...")
        
        try:
            # Start bot manager
            await self.bot_manager.start()
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("⚠️ Received interrupt signal...")
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("🛑 Shutting down bot...")
        self.running = False
        
        if self.bot_manager:
            await self.bot_manager.shutdown()
        
        logger.info("👋 Bot stopped successfully!")
    
    def handle_signal(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}")
        self.running = False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Ultimate Bybit Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --preset scalping --trailing-stop tight
  python main.py --strategy scalping momentum --trail-distance 0.015
  python main.py --preset conservative --breakeven --partial-close
        """
    )
    
    # Strategy configuration
    strategy_group = parser.add_argument_group('Strategy Configuration')
    strategy_group.add_argument(
        '--preset',
        type=str,
        choices=['optimized_multi','scalping','advanced_scalping', 'momentum', 'conservative', 'aggressive', 'ml_only', 'mean_reversion'],
        help='Strategy preset to use'
    )
    strategy_group.add_argument(
        '--strategy',
        type=str,
        nargs='+',
        choices=['optimized_multi','scalping','advanced_scalping', 'momentum', 'mean_reversion', 'ml'],
        help='Specific strategies to enable (overrides preset)'
    )
    
    # Trailing stop configuration
    trailing_group = parser.add_argument_group('Trailing Stop Configuration')
    trailing_group.add_argument(
        '--trailing-stop',
        type=str,
        choices=['tight', 'normal', 'wide', 'atr_based', 'dynamic', 'none'],
        default='normal',
        help='Trailing stop configuration (default: normal)'
    )
    trailing_group.add_argument(
        '--trail-distance',
        type=float,
        help='Override trailing stop distance (e.g., 0.02 for 2%%)'
    )
    trailing_group.add_argument(
        '--breakeven',
        action='store_true',
        help='Enable breakeven stop'
    )
    trailing_group.add_argument(
        '--partial-close',
        action='store_true',
        help='Enable partial position closing'
    )
    
    # Trading mode
    mode_group = parser.add_argument_group('Trading Mode')
    mode_group.add_argument(
        '--testnet',
        action='store_true',
        default=True,
        help='Use testnet (default: True)'
    )
    mode_group.add_argument(
        '--live',
        action='store_true',
        help='Enable live trading (use with caution!)'
    )
    
    # Additional options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Build custom configuration from arguments
    custom_config = {}
    if args.trail_distance:
        custom_config['trail_distance'] = args.trail_distance
    if args.breakeven:
        custom_config['breakeven'] = True
    if args.partial_close:
        custom_config['partial_close'] = True
    if args.strategy:
        custom_config['strategies'] = args.strategy
    
    # Determine trading mode
    mode = 'LIVE TRADING' if args.live else 'TESTNET'
    
    # Print banner
    print(f"""
    ╔══════════════════════════════════════════╗
    ║     ULTIMATE BYBIT TRADING BOT v1.0      ║
    ║         Powered by Advanced AI           ║
    ║                                          ║
    ║  Mode: {mode:34} ║
    ║  Preset: {(args.preset or 'DEFAULT'):32} ║
    ║  Trailing: {args.trailing_stop.upper():30} ║
    ╚══════════════════════════════════════════╝
    """)
    
    # Print configuration summary
    if args.strategy:
        print(f"    Strategies: {', '.join(args.strategy)}")
    if args.trail_distance:
        print(f"    Trail Distance: {args.trail_distance:.1%}")
    if args.breakeven:
        print(f"    Breakeven: ENABLED")
    if args.partial_close:
        print(f"    Partial Close: ENABLED")
    print("")
    
    # Confirm live trading
    if args.live:
        print("⚠️  WARNING: LIVE TRADING MODE ENABLED!")
        print("⚠️  Real money will be at risk!")
        response = input("Type 'YES' to confirm live trading: ")
        if response != 'YES':
            print("Live trading cancelled.")
            return
    
    # Create and start bot
    bot = TradingBot(
        preset=args.preset,
        trailing_stop=args.trailing_stop,
        custom_config=custom_config
    )
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, bot.handle_signal)
    signal.signal(signal.SIGTERM, bot.handle_signal)
    
    # Start bot
    await bot.start()

if __name__ == "__main__":
    # Print Python version for debugging
    import platform
    logger.info(f"Python version: {platform.python_version()}")
    
    # Run the bot
    asyncio.run(main())