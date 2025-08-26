"""
Technical Indicators Module
"""

import numpy as np
import pandas as pd
import talib
from typing import Optional

class TechnicalIndicators:
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator"""
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=period)
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator"""
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'].values,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(
            df['close'].values,
            timeperiod=period,
            nbdevup=std,
            nbdevdn=std
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """Add multiple moving averages"""
        for period in periods:
            df[f'ma_{period}'] = talib.SMA(df['close'].values, timeperiod=period)
        return df
    
    @staticmethod
    def add_volume_profile(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume profile indicators"""
        df['volume_ma'] = talib.SMA(df['volume'].values, timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        df['atr'] = talib.ATR(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=period
        )
        return df
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, fastk: int = 14, slowk: int = 3, slowd: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        slowk, slowd = talib.STOCH(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            fastk_period=fastk,
            slowk_period=slowk,
            slowd_period=slowd
        )
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        return df