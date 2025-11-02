# -*- coding: utf-8 -*-
# ==========================================================
# Strategy Name: NFI SuperTrend Dynamic Dual-Position Strategy (3x Leverage)
# Description:
#  - Uses RSI + EMA(9/21) + SuperTrend to determine market direction
#  - Fixed entry amount: $20 per trade (approx. $3000 total capital)
#  - Averaging logic: Adds position every 5% against price movement, up to 5 times
#  - Each rebuy increases take-profit target by +$0.5
#  - Fixed leverage: 3x
#  - No stoploss (risk is managed through take-profit and rebuy system)
# ==========================================================

from freqtrade.strategy import IStrategy
import pandas as pd
import talib.abstract as ta


class NFI_SuperTrend_Freqtrade_3x(IStrategy):
    # === Basic Configuration ===
    timeframe = '5m'       # Using 5-minute candles
    can_short = True        # Enable short trades (allow long & short simultaneously)
    startup_candle_count = 50
    leverage = 3            # Fixed 3x leverage

    # === Fixed Parameters ===
    stake_amount = 20       # Each trade uses $20
    max_rebuys = 5          # Maximum 5 rebuys
    rebuy_distance = 0.05   # Each rebuy triggered at 5% adverse move
    take_profit_base = 0.5  # Initial take-profit target ($0.5), increases with each rebuy

    # =============================
    #       Indicator Calculation
    # =============================
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Calculate RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Calculate EMA(9) and EMA(21)
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)

        # Calculate SuperTrend
        period = 10
        multiplier = 3.0
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=period)
        dataframe['upperband'] = hl2 + (multiplier * dataframe['atr'])
        dataframe['lowerband'] = hl2 - (multiplier * dataframe['atr'])
        dataframe['in_uptrend'] = True

        # SuperTrend Logic
        for current in range(1, len(dataframe.index)):
            prev = current - 1
            if dataframe['close'][current] > dataframe['upperband'][prev]:
                dataframe.at[current, 'in_uptrend'] = True
            elif dataframe['close'][current] < dataframe['lowerband'][prev]:
                dataframe.at[current, 'in_uptrend'] = False
            else:
                dataframe.at[current, 'in_uptrend'] = dataframe['in_uptrend'][prev]
                if dataframe['in_uptrend'][current] and dataframe['lowerband'][current] < dataframe['lowerband'][prev]:
                    dataframe.at[current, 'lowerband'] = dataframe['lowerband'][prev]
                if not dataframe['in_uptrend'][current] and dataframe['upperband'][current] > dataframe['upperband'][prev]:
                    dataframe.at[current, 'upperband'] = dataframe['upperband'][prev]

        # Signal strength (used for dynamic dual entries)
        dataframe['signal_strength'] = abs(dataframe['rsi'] - 50)

        return dataframe

    # =============================
    #       Entry Conditions
    # =============================
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Long Entry Signal
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema9']) &
                (dataframe['ema9'] > dataframe['ema21']) &
                (dataframe['rsi'] < 40) &
                (dataframe['in_uptrend'] == True)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'long_signal')

        # Short Entry Signal
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['ema9']) &
                (dataframe['ema9'] < dataframe['ema21']) &
                (dataframe['rsi'] > 60) &
                (dataframe['in_uptrend'] == False)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'short_signal')

        return dataframe

    # =============================
    #       Exit (Take-Profit) Logic
    # =============================
    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Dynamic take-profit logic:
        - Take-profit increases with each rebuy (+0.5U per rebuy)
        - No stoploss (ignore negative PnL)
        """
        rebuy_count = trade.user_data.get('rebuys', 0) if trade.user_data else 0
        take_profit_target = self.take_profit_base * (1 + rebuy_count)

        # Close trade when target profit in USD is reached
        if current_profit * self.stake_amount * self.leverage >= take_profit_target:
            return 'take_profit'
        return None

    # =============================
    #       Rebuy Price Logic
    # =============================
    def custom_entry_price(self, pair: str, current_time, proposed_rate, entry_tag, side, **kwargs):
        """
        Defines price for next rebuy:
        - Every 5% against previous entry
        """
        if side == 'long':
            rebuy_price = proposed_rate * (1 - self.rebuy_distance)
        else:
            rebuy_price = proposed_rate * (1 + self.rebuy_distance)
        return rebuy_price

    # =============================
    #       Auto Rebuy Logic
    # =============================
    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Automatically adds position when price moves 5% against direction:
        - Maximum 5 rebuys
        - Each $20
        """
        if trade.user_data is None:
            trade.user_data = {'rebuys': 0}

        rebuy_count = trade.user_data['rebuys']
        if rebuy_count < self.max_rebuys:
            last_entry_price = trade.open_rate
            if trade.is_long and current_rate <= last_entry_price * (1 - self.rebuy_distance):
                trade.user_data['rebuys'] += 1
                return self.stake_amount
            elif trade.is_short and current_rate >= last_entry_price * (1 + self.rebuy_distance):
                trade.user_data['rebuys'] += 1
                return self.stake_amount
        return None

    # =============================
    #       Disable Stoploss
    # =============================
    stoploss = None          # No stoploss
    use_custom_stoploss = False
    trailing_stop = False

    # =============================
    #       Dynamic Dual Entry Logic
    # =============================
    def filter_top_signals(self, dataframe: pd.DataFrame):
        """
        Dynamic dual-entry (optional):
        - Sorts by signal strength
        - Prioritize strongest signals
        """
        top_signals = dataframe.sort_values('signal_strength', ascending=False).head(5)
        return top_signals
