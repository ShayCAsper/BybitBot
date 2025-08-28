# tools/monitor.py
#!/usr/bin/env python3
"""
Trading Bot Monitor (no bot changes required)

- Reads live account + positions from your ExchangeClient
- Parses logs/bot.log to track signals/trades per strategy
- Renders a colored dashboard in the console
"""

import asyncio
import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dotenv import load_dotenv

# Color on Windows consoles
try:
    import colorama
    colorama.just_fix_windows_console()
    from colorama import Fore, Back, Style
except Exception:
    class _Dummy:  # fallback if colorama missing
        def __getattr__(self, _): return ""
    Fore = Back = Style = _Dummy()

# Make local imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from config.preset_loader import apply_preset
from config.config_manager import ConfigManager
from core.exchange_client import ExchangeClient  # your async client
from utils.logger import setup_logger

LOG_PATH = ROOT / "logs" / "bot.log"


class Monitor:
    def __init__(self, preset: str | None, testnet: bool | None, refresh: int = 5):
        self.refresh = max(2, int(refresh))
        self.logger = setup_logger()
        load_dotenv()

        # Optional preset / testnet to match the bot
        if preset:
            try:
                from config.preset_loader import apply_preset as _apply
                _apply(preset)
                self.logger.info(f"[monitor] Applied preset: {preset}")
            except Exception as e:
                self.logger.warning(f"[monitor] Could not apply preset '{preset}': {e}")

        if testnet is True:
            os.environ["TESTNET"] = "True"
        elif testnet is False:
            os.environ["TESTNET"] = "False"

        # State
        self.start_time = datetime.now()
        self.start_balance = 0.0
        self.peak_balance = 0.0
        self.lowest_balance = float("inf")
        self.pnl_history = deque(maxlen=2000)

        # Per-strategy tallies inferred from logs
        self.strategy_metrics = defaultdict(lambda: {
            "signals": 0, "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0
        })
        # Map symbol -> last known strategy (from “Placed <strategy> …” logs)
        self.symbol_strategy = {}

        # For log parsing
        self._last_log_size = 0

        # Exchange client
        self.cfg = ConfigManager()
        self.exchange: ExchangeClient | None = None

    # ---------- helpers ----------

    def _clear(self):
        os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def _fmt_money(x: float) -> str:
        return f"${x:,.2f}"

    @staticmethod
    def _fmt_signed(x: float) -> str:
        return f"{x:+.2f}"

    @staticmethod
    def _color(v: float) -> str:
        if v > 0: return Fore.GREEN
        if v < 0: return Fore.RED
        return Fore.YELLOW

    async def _connect(self):
        self.exchange = ExchangeClient(config=self.cfg)
        await self.exchange.connect()
        # initialize balances
        bal = await self._get_balance()
        # If we still couldn't fetch, leave start at 0 (will self-correct in _draw)
        self.start_balance = bal if bal > 0 else 0.0
        self.peak_balance = self.start_balance
        self.lowest_balance = self.start_balance if self.start_balance > 0 else float("inf")


    async def _get_balance(self) -> float:
        """
        Try multiple sources/shapes:
        1) Your ExchangeClient.get_balance() if it returns a number
        2) ccxt fetch_balance(): prefer total.USDT, else free+used, else Bybit 'info' totalEquity
        3) Last 'Balance: $...' line from logs as a final fallback
        """
        # 1) Your helper
        try:
            if hasattr(self.exchange, "get_balance"):
                v = await self.exchange.get_balance()
                if v is not None:
                    return float(v)
        except Exception:
            pass

        # 2) ccxt
        try:
            fb = await self.exchange.exchange.fetch_balance(params={"type": "swap"})
            # Typical ccxt normalized shape
            total = fb.get("total") or {}
            usdt_total = total.get("USDT")
            if usdt_total is not None:
                return float(usdt_total)

            # Sometimes only "free"/"used" exist
            usdt_free = (fb.get("free") or {}).get("USDT")
            usdt_used = (fb.get("used") or {}).get("USDT")
            if usdt_free is not None or usdt_used is not None:
                return float(usdt_free or 0.0) + float(usdt_used or 0.0)

            # Bybit-specific: try peeking into info blob (unified account)
            info = fb.get("info") or {}
            # v5: info.result.list[0].totalEquity or totalWalletBalance
            res = info.get("result") or {}
            lst = res.get("list") or []
            if lst and isinstance(lst, list):
                row = lst[0] or {}
                for key in ("totalEquity", "totalWalletBalance", "accountBalance", "equity"):
                    if row.get(key) is not None:
                        return float(row[key])

            # Sum all coin totals if nothing else (rare fallback)
            if isinstance(total, dict) and total:
                try:
                    return float(sum(float(x or 0.0) for x in total.values()))
                except Exception:
                    pass
        except Exception:
            pass

        # 3) Parse last Balance from logs (final fallback)
        try:
            if LOG_PATH.exists():
                import re
                last = ""
                with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f.readlines()[-2000:]:
                        if "Balance:" in line:
                            last = line.strip()
                if last:
                    m = re.search(r"Balance:\s*\$([\d,\.]+)", last)
                    if m:
                        return float(m.group(1).replace(",", ""))
        except Exception:
            pass

        return 0.0

    async def _spread_pct(self, sym: str):
        try:
            ob = await self.exchange.fetch_order_book(sym, limit=5)
            bid = ob.get("bids", [[None]])[0][0]
            ask = ob.get("asks", [[None]])[0][0]
            if bid and ask and bid > 0:
                return (ask - bid) / bid * 100.0
        except Exception:
            pass
        return None
        
    async def _get_positions(self):
        try:
            # Prefer your helper if it exists
            if hasattr(self.exchange, "get_positions"):
                return await self.exchange.get_positions()
        except Exception:
            pass
        # Fallback to ccxt unified positions
        try:
            return await self.exchange.exchange.fetch_positions(params={"category": "linear"})
        except Exception as e:
            self.logger.warning(f"[monitor] fetch_positions failed: {e}")
            return []

    async def _ticker(self, symbol: str):
        try:
            return await self.exchange.exchange.fetch_ticker(symbol)
        except Exception:
            return {}

    # ---------- log parsing ----------

    def _tail_new_lines(self) -> list[str]:
        """Read new lines since last read (cheap incremental tail)."""
        if not LOG_PATH.exists():
            return []
        try:
            size = LOG_PATH.stat().st_size
            mode = "r"  # text is fine
            with open(LOG_PATH, mode, encoding="utf-8", errors="ignore") as f:
                if size < self._last_log_size:
                    # log rotated; read from start
                    f.seek(0)
                else:
                    f.seek(self._last_log_size)
                data = f.read()
                self._last_log_size = size
            return data.splitlines()
        except Exception:
            # On any error, just read last 2000 lines
            try:
                with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                    return f.readlines()[-2000:]
            except Exception:
                return []

    def _parse_logs(self):
        """
        Update per-strategy tallies from recent log lines.
        We look for patterns you already log, e.g.:
          - "StrategyManager:generate_proposals -  scalping: 2 proposals -> ..."
          - "✅ Placed scalping BUY SOL/USDT:USDT qty=... (order ...)"
          - "Position closed: BTC/USDT:USDT"
        """
        lines = self._tail_new_lines()
        if not lines:
            return

        # Regexes
        prop_re = re.compile(r"generate_proposals\s+-\s+(\w+):\s+(\d+)\s+proposals", re.I)
        placed_re = re.compile(r"Placed\s+(\w+)\s+(BUY|SELL)\s+([A-Z]+/USDT:USDT)", re.I)
        closed_re = re.compile(r"Position closed:\s+([A-Z]+/USDT:USDT)", re.I)
        win_re = re.compile(r"(TP|TakeProfit)\s+hit", re.I)
        lose_re = re.compile(r"(SL|Stop Loss)\s+hit", re.I)

        for line in lines:
            m = prop_re.search(line)
            if m:
                stg, num = m.group(1).lower(), int(m.group(2))
                self.strategy_metrics[stg]["signals"] += num
                continue

            m = placed_re.search(line)
            if m:
                stg, _side, sym = m.group(1).lower(), m.group(2).upper(), m.group(3)
                self.strategy_metrics[stg]["trades"] += 1
                self.symbol_strategy[sym] = stg
                continue

            m = closed_re.search(line)
            if m:
                sym = m.group(1)
                # If we knew which strategy opened it, increment a completed-trade bucket
                stg = self.symbol_strategy.get(sym)
                if stg:
                    # We don't have realized PnL value in logs, so win/loss
                    # hints (TP/SL) help approximate.
                    if win_re.search(line):
                        self.strategy_metrics[stg]["wins"] += 1
                    elif lose_re.search(line):
                        self.strategy_metrics[stg]["losses"] += 1
                    else:
                        # unknown outcome -> don't count as win/loss
                        pass

    # ---------- rendering ----------

    async def _draw(self):
        self._clear()

        now = datetime.now()
        uptime = now - self.start_time
        h = int(uptime.total_seconds() // 3600)
        m = int((uptime.total_seconds() % 3600) // 60)

        bal = await self._get_balance()

        # If we started with 0 (API race), correct start at first nonzero balance
        if self.start_balance <= 0 and bal > 0:
            self.start_balance = bal
            self.peak_balance = max(self.peak_balance, bal)
            self.lowest_balance = min(self.lowest_balance, bal)

        # Track peak/valley
        if bal > 0:
            self.peak_balance = max(self.peak_balance or bal, bal)
            self.lowest_balance = min(self.lowest_balance if self.lowest_balance != float("inf") else bal, bal)

        init = self.start_balance if self.start_balance > 0 else bal
        pnl = bal - init if init else 0.0
        roi = (pnl / init * 100) if init else 0.0
        dd = ((self.peak_balance - bal) / self.peak_balance * 100) if self.peak_balance else 0.0
        pace_hr = pnl / (uptime.total_seconds() / 3600) if uptime.total_seconds() > 60 else 0.0

        pos = await self._get_positions()
        # Accept both ccxt/unified shapes; consider > 0 contracts/size as open
        def _size_of(p):
            for k in ("contracts", "size", "positionAmt", "contractSize"):
                v = p.get(k) or (p.get("info") or {}).get(k)
                if v is not None:
                    try:
                        return abs(float(v))
                    except Exception:
                        pass
            return 0.0

        open_pos = [p for p in pos if _size_of(p) > 0.0]

        # Header
        print(f"{Back.BLUE}{Fore.WHITE}{' ' * 2}TRADING BOT MONITOR{Style.RESET_ALL}  {now:%Y-%m-%d %H:%M:%S}")
        print(f"{'-'*80}")

        # Account + Perf
        print(f"{Fore.CYAN}ACCOUNT{Style.RESET_ALL}")
        print(f" Start:   {self._fmt_money(init or 0.0)}")
        print(f" Balance: {self._fmt_money(bal)}")
        print(f" Uptime:  {h}h {m}m")
        print(f" P&L:     {self._color(pnl)}{self._fmt_money(pnl)}{Style.RESET_ALL}  "
              f"ROI: {self._color(roi)}{roi:+.2f}%{Style.RESET_ALL}  "
              f"Pace: {self._color(pace_hr)}{self._fmt_signed(pace_hr)}/hr{Style.RESET_ALL}  "
              f"DD: {self._color(-dd)}{dd:.1f}%{Style.RESET_ALL}")
        print("")

        # Positions
        print(f"{Fore.YELLOW}OPEN POSITIONS ({len(open_pos)}){Style.RESET_ALL}")
        if not open_pos:
            print("  (none)")
        else:
            print(f"  {'Symbol':<13}{'Side':<6}{'Size':>10}  {'Entry':>10}  {'Mark':>10}  {'P&L':>12}  {'%':>7}  Strategy")
            print("  " + "-"*76)
            total_unreal = 0.0
            for p in open_pos:
                sym = p.get("symbol", "")
                side = (p.get("side") or "").lower()
                size = _size_of(p)
                entry = float(p.get("entryPrice") or p.get("entry_price") or (p.get("info") or {}).get("avgPrice") or 0.0)
                mark = float(p.get("markPrice") or p.get("mark_price") or 0.0)
                if not mark:
                    try:
                        t = await self._ticker(sym)
                        mark = float(t.get("last") or 0.0)
                    except Exception:
                        mark = entry
                pnl_u = float(p.get("unrealizedPnl") or p.get("unrealized_pnl") or (mark-entry)*size)
                pct = ((mark - entry) / entry * 100) if entry else 0.0
                if side in ("sell", "short"):
                    pct = -pct
                    pnl_u = -pnl_u
                total_unreal += pnl_u
                stg = self.symbol_strategy.get(sym, "-")
                print(f"  {sym:<13}{side:<6}{size:>10.4f}  {entry:>10.2f}  {mark:>10.2f}  "
                      f"{self._color(pnl_u)}{pnl_u:>+12.2f}{Style.RESET_ALL}  {self._color(pct)}{pct:>+6.2f}%{Style.RESET_ALL}  {stg}")
            print("  " + "-"*76)
            print(f"  Unrealized P&L: {self._color(total_unreal)}{total_unreal:+.2f}{Style.RESET_ALL}")
        print("")

        # Market snapshot (BTC/ETH/SOL)
        print(f"{Fore.GREEN}MARKET SNAPSHOT (24h){Style.RESET_ALL}")
        print(f"  {'Coin':<6}{'Price':>12}  {'%':>7}  {'Volume':>12}  {'Spread':>7}  Trend")
        for coin in ("BTC", "ETH", "SOL"):
            sym = f"{coin}/USDT:USDT"
            t = await self._ticker(sym)
            last = float(t.get("last") or 0.0)
            pct = float(t.get("percentage") or 0.0)
            vol = float(t.get("quoteVolume") or 0.0)
            bid = t.get("bid")
            ask = t.get("ask")
            spr_pct = await self._spread_pct(sym)
            trend = "Up" if pct > 0 else "Down" if pct < 0 else "Flat"
            col = Fore.GREEN if pct > 0 else Fore.RED if pct < 0 else Fore.YELLOW
            spr_txt = f"{spr_pct:>6.3f}%" if spr_pct is not None else "   —  "
            print(f"  {coin:<6}{last:>12.2f}  {col}{pct:>+6.2f}%{Style.RESET_ALL}  {vol/1_000_000:>9.1f}M  {spr_txt}  {trend}")
        print("")

        # Strategy performance (from logs)
        print(f"{Fore.MAGENTA}STRATEGY PERFORMANCE (from logs){Style.RESET_ALL}")
        print(f"  {'Strategy':<16}{'Signals':>8}  {'Trades':>8}  {'Win%':>7}  {'PnL*':>10}")
        for stg, m in sorted(self.strategy_metrics.items()):
            win_rate = (m["wins"] / m["trades"] * 100) if m["trades"] else 0.0
            pnl_stg = m["pnl"]
            print(f"  {stg:<16}{m['signals']:>8}  {m['trades']:>8}  {win_rate:>6.1f}%  {self._color(pnl_stg)}{pnl_stg:>+10.2f}{Style.RESET_ALL}")

        # Footer
        mode = "TESTNET" if os.getenv("TESTNET", "True").lower() in ("1","true","yes","y") else "LIVE"
        print("\n" + "-"*80)
        print(f"Refresh: {self.refresh}s   Mode: {mode}   Ctrl+C to exit")

    # ---------- main loop ----------

    async def run(self):
        await self._connect()
        # initial parse so we have some metrics
        self._parse_logs()
        try:
            while True:
                self._parse_logs()
                await self._draw()
                await asyncio.sleep(self.refresh)
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Monitor stopped by user{Style.RESET_ALL}")
        finally:
            try:
                await self.exchange.exchange.close()
            except Exception:
                pass


def parse_args():
    p = argparse.ArgumentParser(description="Trading Bot Monitor")
    p.add_argument("--preset", default=None, help="apply a preset before connecting (optional)")
    p.add_argument("--testnet", dest="testnet", action="store_true", help="force TESTNET")
    p.add_argument("--live", dest="testnet", action="store_false", help="force LIVE")
    p.add_argument("--refresh", type=int, default=5, help="refresh seconds (default: 5)")
    p.set_defaults(testnet=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(Monitor(preset=args.preset, testnet=args.testnet, refresh=args.refresh).run())
