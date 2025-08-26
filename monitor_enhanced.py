import asyncio
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.append(str(Path(__file__).parent))

from core.exchange_client import BybitClient
from config.config_manager import ConfigManager

load_dotenv()

async def enhanced_monitor():
    """Enhanced monitoring with strategy performance tracking"""
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    client = BybitClient(config['exchange'])
    await client.connect()
    
    # Track metrics
    metrics = defaultdict(lambda: {'signals': 0, 'trades': 0, 'pnl': 0})
    start_balance = await client.get_balance()
    start_time = datetime.now()
    
    print("Enhanced Bot Monitor - Press Ctrl+C to stop\n")
    
    while True:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            current_time = datetime.now()
            runtime = current_time - start_time
            
            print("="*70)
            print(f"🤖 BOT PERFORMANCE MONITOR - {current_time.strftime('%H:%M:%S')}")
            print(f"Runtime: {runtime.total_seconds()/3600:.1f} hours")
            print("="*70)
            
            # Account info
            current_balance = await client.get_balance()
            total_pnl = current_balance - start_balance
            pnl_pct = (total_pnl / start_balance * 100) if start_balance > 0 else 0
            
            print(f"\n💰 ACCOUNT:")
            print(f"  Balance: ${current_balance:,.2f}")
            print(f"  Total P&L: ${total_pnl:+,.2f} ({pnl_pct:+.2f}%)")
            print(f"  Hourly Rate: ${total_pnl / (runtime.total_seconds()/3600):+,.2f}/hr" if runtime.total_seconds() > 0 else "")
            
            # Positions
            positions = await client.get_positions()
            open_positions = [p for p in positions if p.get('contracts', 0) > 0]
            
            if open_positions:
                print(f"\n📊 POSITIONS ({len(open_positions)}):")
                total_unrealized = 0
                for pos in open_positions:
                    symbol = pos.get('symbol', '').split('/')[0]
                    side = pos.get('side', '')
                    size = pos.get('contracts', 0)
                    entry = pos.get('entryPrice', 0)
                    mark = pos.get('markPrice', 0)
                    pnl = pos.get('unrealizedPnl', 0) or 0
                    total_unrealized += pnl
                    
                    pnl_pct = ((mark - entry) / entry * 100) if entry > 0 else 0
                    if side == 'short':
                        pnl_pct = -pnl_pct
                    
                    emoji = "🟢" if pnl >= 0 else "🔴"
                    print(f"  {emoji} {symbol}: {side.upper()} {size} @ ${entry:,.0f} → ${mark:,.0f} | PNL: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                
                print(f"  Total Unrealized: ${total_unrealized:+,.2f}")
            
            # Market conditions
            print(f"\n📈 MARKET CONDITIONS:")
            for coin in ['BTC', 'ETH', 'SOL']:
                try:
                    symbol = f"{coin}/USDT:USDT"
                    ticker = await client.exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    change_pct = ticker.get('percentage', 0)
                    volume = ticker.get('quoteVolume', 0) / 1e6  # In millions
                    
                    trend = "📈" if change_pct > 1 else "📉" if change_pct < -1 else "➡️"
                    print(f"  {coin}: ${price:,.0f} ({change_pct:+.1f}%) Vol: ${volume:.1f}M {trend}")
                except:
                    pass
            
            # Strategy status
            print(f"\n⚡ STRATEGIES:")
            print(f"  Active: Scalping, Momentum, Mean Reversion")
            print(f"  Trading: {'ENABLED ✅' if os.getenv('ENABLE_TRADING', 'False').lower() == 'true' else 'DISABLED ❌'}")
            print(f"  Signal Check: Every 30 seconds")
            
            print("\n" + "="*70)
            
            await asyncio.sleep(5)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(5)
    
    await client.disconnect()

asyncio.run(enhanced_monitor())