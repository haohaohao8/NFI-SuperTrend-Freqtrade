# dual_direction_dca_futures.py
# =====================================================
# 适用于 Binance Futures (USDT-M)，多空双向 DCA 策略
# SuperTrend + RSI 自动方向判断 + 分层止盈
# =====================================================

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
import pandas as pd
import numpy as np
import talib.abstract as ta

# ======================== 参数区 ========================

# 多头补仓逻辑（下跌）
LONG_FIRST_ADD_THRESHOLD = -0.5      # 首次补仓亏 0.5 U
LONG_SECOND_ADD_THRESHOLD = -2       # 第二次补仓亏 2 U
LONG_NEXT_ADD_THRESHOLD_PCT = -0.05  # 后续补仓下跌 5% 且止跌

# 空头补仓逻辑（上涨）
SHORT_FIRST_ADD_THRESHOLD = 0.5      # 首次补仓回吐 0.5 U
SHORT_SECOND_ADD_THRESHOLD = 2       # 第二次补仓回吐 2 U
SHORT_NEXT_ADD_THRESHOLD_PCT = 0.05  # 后续补仓上涨 5% 且止涨

POSITION_SIZE = 20                   # 每笔仓位 20 USDT
MAX_DCA_LAYERS = 6                   # 最多 6 层补仓
MAX_ACTIVE_PAIRS = 12                # 最多同时开仓 12 个币

# 止盈表（按补仓层数增加）
TAKE_PROFIT_TABLE = {
    1: 0.5, 2: 1, 3: 1.5, 4: 2, 5: 2.5, 6: 3
}

PAIR_LIST = [
    "SOL/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOGE/USDT:USDT",
    "HYPE/USDT:USDT", "XRP/USDT:USDT", "ENA/USDT:USDT", "ARB/USDT:USDT",
    "APT/USDT:USDT", "LDO/USDT:USDT", "OP/USDT:USDT", "PUMP/USDT:USDT",
    "SUI/USDT:USDT", "LINK/USDT:USDT", "VIRTUALS/USDT:USDT", "FARTCOIN/USDT:USDT",
    "JUP/USDT:USDT", "AAVE/USDT:USDT", "ONDO/USDT:USDT", "EIGEN/USDT:USDT",
    "BIO/USDT:USDT", "DOT/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT"
]

# =========================================================

class DualDirectionalDCAFutures(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    minimal_roi = {}
    stoploss = -0.99
    use_custom_stoploss = False
    process_only_new_candles = True
    can_short = True     # 🔥 启用双向交易

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df['rsi'] = ta.RSI(df, timeperiod=14)
        st = self.supertrend(df)
        df['supertrend_direction'] = st['STX']
        return df

    @staticmethod
    def supertrend(df, period=10, multiplier=3):
        hl2 = (df['high'] + df['low']) / 2
        atr = ta.ATR(df, timeperiod=period)
        final_upperband = hl2 + multiplier * atr
        final_lowerband = hl2 - multiplier * atr
        trend = [1]
        for i in range(1, len(df)):
            if df['close'][i] > final_upperband[i-1]:
                trend.append(1)
            elif df['close'][i] < final_lowerband[i-1]:
                trend.append(-1)
            else:
                trend.append(trend[-1])
        return pd.DataFrame({'STX': trend}, index=df.index)

    # 自动判定方向
    def should_open_trade(self, pair, df):
        last = df.iloc[-1]
        can_long = last['supertrend_direction'] == 1 and 30 < last['rsi'] < 70
        can_short = last['supertrend_direction'] == -1 and 30 < last['rsi'] < 70
        return 'long' if can_long else ('short' if can_short else None)

    # 补仓逻辑
    def should_dca(self, trade: Trade, df: pd.DataFrame) -> bool:
        layer = len(trade.orders)
        current_price = df.iloc[-1]['close']
        direction = 'long' if trade.is_long else 'short'
        avg_price = trade.open_rate
        profit = trade.calc_profit_ratio(current_price)

        if direction == 'long':
            if layer == 1 and profit <= LONG_FIRST_ADD_THRESHOLD / POSITION_SIZE:
                return True
            elif layer == 2 and profit <= LONG_SECOND_ADD_THRESHOLD / POSITION_SIZE:
                return True
            elif 3 <= layer < MAX_DCA_LAYERS:
                prev_add_price = trade.orders[-1].rate
                drop_pct = (current_price - prev_add_price) / prev_add_price
                stop = df.iloc[-1]['supertrend_direction'] == 1
                if drop_pct <= LONG_NEXT_ADD_THRESHOLD_PCT and stop:
                    return True
        else:
            if layer == 1 and profit >= SHORT_FIRST_ADD_THRESHOLD / POSITION_SIZE:
                return True
            elif layer == 2 and profit >= SHORT_SECOND_ADD_THRESHOLD / POSITION_SIZE:
                return True
            elif 3 <= layer < MAX_DCA_LAYERS:
                prev_add_price = trade.orders[-1].rate
                up_pct = (current_price - prev_add_price) / prev_add_price
                stop = df.iloc[-1]['supertrend_direction'] == -1
                if up_pct >= SHORT_NEXT_ADD_THRESHOLD_PCT and stop:
                    return True
        return False

    # 止盈逻辑
    def custom_exit(self, trade: Trade, df: pd.DataFrame):
        layer = len(trade.orders)
        current_price = df.iloc[-1]['close']
        target_profit_u = TAKE_PROFIT_TABLE.get(layer, 0.5)
        base_price = np.average([o.rate for o in trade.orders])
        direction = 1 if trade.is_long else -1
        profit_u = (current_price - base_price) * trade.amount * direction
        if profit_u >= target_profit_u:
            return True
        return False

    def confirm_trade_entry(self, pair, trade_type, current_total_trades):
        return current_total_trades < MAX_ACTIVE_PAIRS

    def populate_buy_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df['buy'] = 0
        if self.should_open_trade(metadata['pair'], df) == 'long':
            df.iloc[-1, df.columns.get_loc('buy')] = 1
        return df

    def populate_sell_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df['sell'] = 0
        if self.should_open_trade(metadata['pair'], df) == 'short':
            df.iloc[-1, df.columns.get_loc('sell')] = 1
        return df

