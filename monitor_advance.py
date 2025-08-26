#!/usr/bin/env python3
"""
Advanced Trading Bot Monitor with Real-Time Dashboard
Shows all relevant trading metrics and performance data
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from colorama import init, Fore, Back, Style
from dotenv import load_dotenv

# Initialize colorama for Windows color support
init()

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.exchange_client import BybitClient
from config.config_manager import ConfigManager
from utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

class TradingMonitor:
    def __init__(self):
        self.config = None
        self.client = None
        self.start_time = datetime.now()
        self.start_balance = 0
        self.peak_balance = 0
        self.lowest_balance = float('inf')
        
        # Performance tracking
        self.trades_history = deque(maxlen=100)
        self.signals_history = deque(maxlen=50)
        self.pnl_history = deque(maxlen=100)
        
        # Strategy metrics
        self.strategy_metrics = {
            'scalping': {'signals': 0, 'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0},
            'momentum': {'signals': 0, 'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0},
            'mean_reversion': {'signals': 0, 'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0},
            'ml': {'signals': 0, 'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0}
        }
        
        # Hourly metrics
        self.hourly_pnl = defaultdict(float)
        self.hourly_trades = defaultdict(int)
        
    async def initialize(self):
        """Initialize connection and get starting metrics"""
        config_manager = ConfigManager()
        self.config = config_manager.load_config()
        
        self.client = BybitClient(self.config['exchange'])
        await self.client.connect()
        
        self.start_balance = await self.client.get_balance()
        self.peak_balance = self.start_balance
        self.lowest_balance = self.start_balance
        
        # Parse historical data from logs if available
        self.parse_log_history()
    
    def parse_log_history(self):
        """Parse recent log files for historical data"""
        try:
            log_path = Path('logs/bot.log')
            if log_path.exists():
                with open(log_path, 'r') as f:
                    lines = f.readlines()[-1000:]  # Last 1000 lines
                    
                for line in lines:
                    # Parse trade executions
                    if 'Trade executed' in line or 'Order placed' in line:
                        self.hourly_trades[datetime.now().hour] += 1
                    
                    # Parse signals
                    if 'Generated' in line and 'signals' in line:
                        try:
                            # Extract number of signals
                            parts = line.split('Generated')
                            if len(parts) > 1:
                                num_str = parts[1].split('signals')[0].strip()
                                num = int(num_str)
                                if 'momentum' in line.lower():
                                    self.strategy_metrics['momentum']['signals'] += num
                                elif 'scalping' in line.lower():
                                    self.strategy_metrics['scalping']['signals'] += num
                                elif 'mean_reversion' in line.lower():
                                    self.strategy_metrics['mean_reversion']['signals'] += num
                        except:
                            pass
        except Exception as e:
            logger.debug(f"Could not parse log history: {e}")
    
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_number(self, num, decimals=2):
        """Format numbers with proper comma separation"""
        if abs(num) >= 1000:
            return f"{num:,.{decimals}f}"
        return f"{num:.{decimals}f}"
    
    def get_color(self, value, threshold=0):
        """Get color based on value"""
        if value > threshold:
            return Fore.GREEN
        elif value < threshold:
            return Fore.RED
        return Fore.YELLOW
    
    async def get_market_summary(self):
        """Get market conditions summary"""
        summary = {}
        
        for coin in ['BTC', 'ETH', 'SOL']:
            try:
                symbol = f"{coin}/USDT:USDT"
                ticker = await self.client.exchange.fetch_ticker(symbol)
                
                summary[coin] = {
                    'price': ticker['last'],
                    'change_24h': ticker.get('percentage', 0),
                    'volume_24h': ticker.get('quoteVolume', 0),
                    'high_24h': ticker.get('high', 0),
                    'low_24h': ticker.get('low', 0),
                    'bid': ticker.get('bid', 0),
                    'ask': ticker.get('ask', 0),
                    'spread': ((ticker.get('ask', 0) - ticker.get('bid', 0)) / ticker.get('bid', 1) * 100) if ticker.get('bid', 0) > 0 else 0
                }
            except Exception as e:
                logger.debug(f"Could not fetch {coin} ticker: {e}")
                summary[coin] = None
        
        return summary
    
    async def display_dashboard(self):
        """Display the main monitoring dashboard"""
        self.clear_screen()
        
        current_time = datetime.now()
        runtime = current_time - self.start_time
        hours_run = runtime.total_seconds() / 3600
        
        # Header
        print(f"{Back.BLUE}{Fore.WHITE}{'═' * 80}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE} 🤖 ADVANCED TRADING BOT MONITOR {current_time.strftime('%Y-%m-%d %H:%M:%S')} {' ' * 13}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}{'═' * 80}{Style.RESET_ALL}")
        
        # Get current data
        current_balance = await self.client.get_balance()
        positions = await self.client.get_positions()
        market_summary = await self.get_market_summary()
        
        # Update peak/lowest
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        if current_balance < self.lowest_balance:
            self.lowest_balance = current_balance
        
        # Calculate metrics
        total_pnl = current_balance - self.start_balance
        pnl_pct = (total_pnl / self.start_balance * 100) if self.start_balance > 0 else 0
        hourly_rate = total_pnl / hours_run if hours_run > 0 else 0
        drawdown = ((self.peak_balance - current_balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0
        
        # ACCOUNT SECTION
        print(f"\n{Fore.CYAN}╔{'═' * 38}╦{'═' * 39}╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║ 💰 ACCOUNT OVERVIEW{' ' * 18}║ 📊 PERFORMANCE METRICS{' ' * 16}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╠{'═' * 38}╬{'═' * 39}╣{Style.RESET_ALL}")
        
        # Left column - Account
        pnl_color = self.get_color(total_pnl)
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Balance: {Fore.WHITE}${self.format_number(current_balance)}{' ' * (27 - len(self.format_number(current_balance)))} "
              f"{Fore.CYAN}║{Style.RESET_ALL} Runtime: {Fore.WHITE}{int(hours_run)}h {int((hours_run % 1) * 60)}m{' ' * 26}║")
        
        # Format P&L with sign
        pnl_str = f"${total_pnl:+,.2f}" if total_pnl != 0 else "$0.00"
        roi_str = f"{pnl_pct:+.2f}%" if pnl_pct != 0 else "0.00%"
        hourly_str = f"${hourly_rate:+.2f}/hr" if hourly_rate != 0 else "$0.00/hr"
        daily_str = f"${hourly_rate * 24:+,.2f}" if hourly_rate != 0 else "$0.00"
        
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Start:   ${self.format_number(self.start_balance)}{' ' * (27 - len(self.format_number(self.start_balance)))} "
              f"{Fore.CYAN}║{Style.RESET_ALL} Total P&L: {pnl_color}{pnl_str}{Style.RESET_ALL}{' ' * (27 - len(pnl_str))}║")
        
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Peak:    ${self.format_number(self.peak_balance)}{' ' * (27 - len(self.format_number(self.peak_balance)))} "
              f"{Fore.CYAN}║{Style.RESET_ALL} ROI:       {pnl_color}{roi_str}{Style.RESET_ALL}{' ' * (27 - len(roi_str))}║")
        
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Lowest:  ${self.format_number(self.lowest_balance)}{' ' * (27 - len(self.format_number(self.lowest_balance)))} "
              f"{Fore.CYAN}║{Style.RESET_ALL} Hourly:    {pnl_color}{hourly_str}{Style.RESET_ALL}{' ' * (25 - len(hourly_str))}║")
        
        drawdown_color = Fore.GREEN if drawdown < 5 else Fore.YELLOW if drawdown < 10 else Fore.RED
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Drawdown: {drawdown_color}{drawdown:.1f}%{Style.RESET_ALL}{' ' * (26 - len(f'{drawdown:.1f}%'))} "
              f"{Fore.CYAN}║{Style.RESET_ALL} Daily Est: {pnl_color}{daily_str}{Style.RESET_ALL}{' ' * (26 - len(daily_str))}║")
        
        print(f"{Fore.CYAN}╚{'═' * 38}╩{'═' * 39}╝{Style.RESET_ALL}")
        
        # POSITIONS SECTION
        open_positions = [p for p in positions if p.get('contracts', 0) > 0]
        
        print(f"\n{Fore.YELLOW}╔{'═' * 78}╗{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}║ 📈 OPEN POSITIONS ({len(open_positions)}/5){' ' * (78 - 23 - len(str(len(open_positions))))}║{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}╠{'═' * 78}╣{Style.RESET_ALL}")
        
        if open_positions:
            print(f"{Fore.YELLOW}║{Style.RESET_ALL} {'Symbol':<8} {'Side':<5} {'Size':<12} {'Entry':<10} {'Current':<10} {'P&L':<12} {'%':<7} ║")
            print(f"{Fore.YELLOW}║{Style.RESET_ALL} {'-' * 76} ║")
            
            total_unrealized = 0
            for pos in open_positions:
                symbol = pos.get('symbol', '').split('/')[0]
                side = pos.get('side', '')
                size = pos.get('contracts', 0)
                entry = pos.get('entryPrice', 0) or pos.get('entry_price', 0)
                mark = pos.get('markPrice', 0) or pos.get('mark_price', 0)
                pnl = pos.get('unrealizedPnl', 0) or pos.get('unrealized_pnl', 0) or 0
                
                if mark == 0:  # If no mark price, try to get current price
                    try:
                        mark = await self.client.get_price(pos.get('symbol', ''))
                    except:
                        mark = entry
                
                pnl_pct = ((mark - entry) / entry * 100) if entry > 0 else 0
                if side.lower() == 'short' or side.lower() == 'sell':
                    pnl_pct = -pnl_pct
                    pnl = -pnl if pnl > 0 else pnl
                
                total_unrealized += pnl
                
                pnl_color = self.get_color(pnl)
                side_color = Fore.GREEN if side.lower() in ['long', 'buy'] else Fore.RED
                
                pnl_str = f"${pnl:+.2f}" if pnl != 0 else "$0.00"
                pnl_pct_str = f"{pnl_pct:+.1f}%" if pnl_pct != 0 else "0.0%"
                
                print(f"{Fore.YELLOW}║{Style.RESET_ALL} {symbol:<8} {side_color}{side[:5]:<5}{Style.RESET_ALL} "
                      f"{size:<12.4f} ${entry:<9.0f} ${mark:<9.0f} "
                      f"{pnl_color}{pnl_str:<12}{Style.RESET_ALL} {pnl_color}{pnl_pct_str:<7}{Style.RESET_ALL} ║")
            
            print(f"{Fore.YELLOW}║{Style.RESET_ALL} {'-' * 76} ║")
            unreal_color = self.get_color(total_unrealized)
            unreal_str = f"${total_unrealized:+.2f}" if total_unrealized != 0 else "$0.00"
            print(f"{Fore.YELLOW}║{Style.RESET_ALL} {'TOTAL UNREALIZED P&L:':<50} {unreal_color}{unreal_str:>15}{Style.RESET_ALL}      ║")
        else:
            print(f"{Fore.YELLOW}║{Style.RESET_ALL} {'No open positions':<76} ║")
        
        print(f"{Fore.YELLOW}╚{'═' * 78}╝{Style.RESET_ALL}")
        
        # MARKET CONDITIONS SECTION
        print(f"\n{Fore.GREEN}╔{'═' * 78}╗{Style.RESET_ALL}")
        print(f"{Fore.GREEN}║ 🌍 MARKET CONDITIONS{' ' * 57}║{Style.RESET_ALL}")
        print(f"{Fore.GREEN}╠{'═' * 78}╣{Style.RESET_ALL}")
        print(f"{Fore.GREEN}║{Style.RESET_ALL} {'Coin':<6} {'Price':<12} {'24h %':<10} {'Volume':<15} {'Spread':<8} {'Trend':<10} ║")
        print(f"{Fore.GREEN}║{Style.RESET_ALL} {'-' * 76} ║")
        
        for coin, data in market_summary.items():
            if data:
                change_color = self.get_color(data['change_24h'])
                volume_m = data['volume_24h'] / 1_000_000
                
                # Determine trend
                if data['change_24h'] > 2:
                    trend = "🔥 Strong Up"
                    trend_color = Fore.GREEN
                elif data['change_24h'] > 0:
                    trend = "📈 Up"
                    trend_color = Fore.GREEN
                elif data['change_24h'] < -2:
                    trend = "💀 Strong Down"
                    trend_color = Fore.RED
                elif data['change_24h'] < 0:
                    trend = "📉 Down"
                    trend_color = Fore.RED
                else:
                    trend = "➡️ Sideways"
                    trend_color = Fore.YELLOW
                
                change_str = f"{data['change_24h']:+.2f}%" if data['change_24h'] != 0 else "0.00%"
                
                print(f"{Fore.GREEN}║{Style.RESET_ALL} {coin:<6} ${data['price']:<11.0f} "
                      f"{change_color}{change_str:>8}{Style.RESET_ALL}   "
                      f"${volume_m:<12.1f}M  {data['spread']:<7.3f}% "
                      f"{trend_color}{trend:<10}{Style.RESET_ALL} ║")
        
        print(f"{Fore.GREEN}╚{'═' * 78}╝{Style.RESET_ALL}")
        
        # STRATEGY PERFORMANCE SECTION
        print(f"\n{Fore.MAGENTA}╔{'═' * 78}╗{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}║ ⚡ STRATEGY PERFORMANCE{' ' * 54}║{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}╠{'═' * 78}╣{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}║{Style.RESET_ALL} {'Strategy':<15} {'Signals':<10} {'Trades':<10} {'Win Rate':<12} {'P&L':<15} ║")
        print(f"{Fore.MAGENTA}║{Style.RESET_ALL} {'-' * 76} ║")
        
        for strategy, metrics in self.strategy_metrics.items():
            if metrics['trades'] > 0:
                win_rate = (metrics['wins'] / metrics['trades'] * 100) if metrics['trades'] > 0 else 0
                win_color = Fore.GREEN if win_rate >= 60 else Fore.YELLOW if win_rate >= 50 else Fore.RED
                pnl_color = self.get_color(metrics['pnl'])
                
                pnl_str = f"${metrics['pnl']:+.2f}" if metrics['pnl'] != 0 else "$0.00"
                
                print(f"{Fore.MAGENTA}║{Style.RESET_ALL} {strategy.capitalize():<15} "
                      f"{metrics['signals']:<10} {metrics['trades']:<10} "
                      f"{win_color}{win_rate:>6.1f}%{Style.RESET_ALL}      "
                      f"{pnl_color}{pnl_str:>11}{Style.RESET_ALL}     ║")
        
        print(f"{Fore.MAGENTA}╚{'═' * 78}╝{Style.RESET_ALL}")
        
        # STATUS BAR
        print(f"\n{Back.BLACK}{Fore.WHITE}{'─' * 80}{Style.RESET_ALL}")
        
        trading_status = "ENABLED ✅" if os.getenv('ENABLE_TRADING', 'False').lower() == 'true' else "DISABLED ❌"
        testnet_status = "TESTNET" if os.getenv('BYBIT_TESTNET', 'True').lower() == 'true' else "MAINNET"
        
        print(f"Trading: {Fore.GREEN if 'ENABLED' in trading_status else Fore.RED}{trading_status}{Style.RESET_ALL} | "
              f"Mode: {Fore.YELLOW}{testnet_status}{Style.RESET_ALL} | "
              f"Refresh: 5s | Press Ctrl+C to exit")
        
        print(f"{Back.BLACK}{Fore.WHITE}{'─' * 80}{Style.RESET_ALL}")
    
    async def run(self):
        """Main monitoring loop"""
        await self.initialize()
        
        print(f"{Fore.CYAN}Initializing Advanced Monitor...{Style.RESET_ALL}")
        await asyncio.sleep(2)
        
        try:
            while True:
                try:
                    await self.display_dashboard()
                    await asyncio.sleep(5)  # Refresh every 5 seconds
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"{Fore.RED}Display error: {e}{Style.RESET_ALL}")
                    await asyncio.sleep(5)
                    
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Monitor stopped by user{Style.RESET_ALL}")
        finally:
            await self.client.disconnect()
            print(f"{Fore.GREEN}Disconnected successfully{Style.RESET_ALL}")

async def main():
    monitor = TradingMonitor()
    await monitor.run()

if __name__ == "__main__":
    # Check if colorama is installed
    try:
        from colorama import init, Fore, Back, Style
    except ImportError:
        print("Installing colorama for colored output...")
        os.system("pip install colorama")
        from colorama import init, Fore, Back, Style
    
    # Run the monitor
    asyncio.run(main())