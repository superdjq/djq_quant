"""
module 'djq_talib' calculates a variety of indicators like 'MA', 'ROC', 'bollinger band'...
with the help of lib talib
"""
import talib as ta
import pandas as pd
import numpy as np
import zsys


def get_all_finanical_indicators(df, freq_lst=(5, 15, 30), divided_by_close=False):
    """
    :param df: instance of pandas.DataFrame, with columns {'open', 'close', 'high', 'low', 'volume'}
    :param freq_lst: a list of indicator parameters which you want to use in your model,
                    the most common parameter like 14, 28, 252...
    :param divided_by_close: Set True to let all price indicators as a pct form of close price
    :return: pandas.DataFrame with all indicators
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('The type of df to be processed should be pandas.DataFrame!')

    df = df.sort_values('date', ascending=True)
    df.loc[df.high==df.low, 'high'] = df.loc[df.high==df.low, 'high'] + 1e-8


    get_talib_indicators(df, freq_lst, divided_by_close=divided_by_close)
    get_last_data(df)
    get_stockchart_indicators(df)


    #Data Wash
    df = df.replace([np.inf, -np.inf], np.nan)
    '''for indicator in zsys.TDS_talib_indicators_all + ['volume']:
        if df[indicator].mean() > 10000:
            tmp = df[indicator].copy()
            print(indicator)
            df[indicator] = (tmp - tmp.mean()) / tmp.std()'''
    '''for indicator in df.columns:
        if df[indicator].dtype != 'O':
            tmp = df[indicator].copy()
            df[indicator] = (tmp - tmp.mean()) / tmp.std()'''



    df = df.round(5)
    df.dropna(inplace=True)


    return df


def calc_hurst(value):
    lags = range(2, 100)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(value[lag:], value[:-lag])))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def get_stockchart_indicators(df):
    """
    Calculate the most common used indicators in financial analysis
    :param df: pandas.DataFrame with {'open', 'close', 'high', 'low', 'volume'} and index == date
    :return: pandas.DataFrame
    """
    # Bollinger Bands
    df['BBANDS_20_UP'], df['BBANDS_20_MID'], df['BBANDS_20_LOW'] = ta.BBANDS(df.close, timeperiod=20, nbdevup=2,
                                                                             nbdevdn=2, matype=0)
    df['%B_20'] = (df.close - df['BBANDS_20_LOW']) / (df['BBANDS_20_UP'] - df['BBANDS_20_LOW'])
    # Chandelier_Exit
    df['Chandelier_Exit_22_long'] = ta.MAX(df.high, 22) - ta.ATR(df.high, df.low, df.close, 22) * 3
    df['Chandelier_Exit_22_short'] = ta.MIN(df.low, 22) + ta.ATR(df.high, df.low, df.close, 22) * 3
    # Ichimoku Clouds
    df['Conversion_Line_9'] = (ta.MAX(df.high, 9) + ta.MIN(df.low, 9)) / 2
    df['Base_Line_26'] = (ta.MAX(df.high, 26) + ta.MIN(df.low, 26)) / 2
    df['Leading_Span_A'] = (df['Conversion_Line_9'] + df['Base_Line_26']) / 2
    df['Leading_Span_B'] = (ta.MAX(df.high, 52) + ta.MIN(df.low, 52)) / 2
    df['Lagging_Span'] = df.close.shift(26)
    # KAMA
    df['KAMA_10'] = ta.KAMA(df.close, timeperiod=10)
    # Keltner Channels
    df['Keltner_Channels_middle'] = ta.EMA(df.close, timeperiod=20)
    df['Keltner_Channels_upper'] = df['Keltner_Channels_middle'] + 2 * ta.ATR(df.high, df.low, df.close, 10)
    df['Keltner_Channels_lower'] = df['Keltner_Channels_middle'] - 2 * ta.ATR(df.high, df.low, df.close, 10)
    # Moving Averages
    df['SMA_20'] = ta.SMA(df.close, timeperiod=20)
    df['SMA_50'] = ta.SMA(df.close, timeperiod=50)
    df['SMA_60'] = ta.SMA(df.close, timeperiod=60)
    df['SMA_150'] = ta.SMA(df.close, timeperiod=150)
    df['EMA_60'] = ta.EMA(df.close, timeperiod=60)
    df['SMA_volume_200'] = ta.SMA(df.volume, timeperiod=200)
    # Moving Average Envelopes
    df['Upper_Envelope_20'] = df['SMA_20'] + df['SMA_20'] * 0.025
    df['Lower_Envelope_20'] = df['SMA_20'] - df['SMA_20'] * 0.025
    df['Upper_Envelope_50'] = df['SMA_50'] + df['SMA_50'] * 0.1
    df['Lower_Envelope_50'] = df['SMA_50'] - df['SMA_50'] * 0.1
    # Parabolic SAR
    df['SAR_0.01_0.2'] = ta.SAR(df.high, df.low, acceleration=0.01, maximum=0.2)
    # Pivot Points
    df['Pivot_Point'] = (df.high + df.low + df.close) / 3
    df['Support_1'] = df['Pivot_Point'] - 0.382 * (df.high - df.low)
    df['Support_2'] = df['Pivot_Point'] - 0.618 * (df.high - df.low)
    df['Support_3'] = df['Pivot_Point'] - 1 * (df.high - df.low)
    df['Resistance_1'] = df['Pivot_Point'] + 0.382 * (df.high - df.low)
    df['Resistance_2'] = df['Pivot_Point'] + 0.618 * (df.high - df.low)
    df['Resistance_3'] = df['Pivot_Point'] + 1 * (df.high - df.low)
    # Price Channels
    df['high_20'] = ta.MAX(df.high, 20)
    df['low_20'] = ta.MIN(df.low, 20)
    df['Price_Channels_center'] = (df['high_20'] + df['low_20']) / 2
    # Volume Weighted Average Price
    VP = df['Pivot_Point'] * df.volume
    VP_total = VP.cumsum()
    V_total = df.volume.cumsum()
    df['VWAP'] = VP_total / V_total
    # Accumulation Distribution Line
    df['ADL'] = (df.volume * (df.close - df.low - df.high + df.close) / (df.high - df.low)).cumsum()
    # Aroon
    df['AROON_DOWN_25'], df['AROON_UP_25'] = ta.AROON(df.high, df.low, timeperiod=25)
    df['Aroon_Oscillator_25'] = df['AROON_UP_25'] - df['AROON_DOWN_25']
    # Average Directional Index (ADX)
    df['MINUS_DI_14'] = ta.MINUS_DI(df.high, df.low, df.close, timeperiod=14)
    df['MINUS_DM_14'] = ta.MINUS_DM(df.high, df.low, timeperiod=14)
    df['PLUS_DI_14'] = ta.PLUS_DI(df.high, df.low, df.close, timeperiod=14)
    df['PLUS_DM_14'] = ta.PLUS_DM(df.high, df.low, timeperiod=14)
    df['ADX_14'] = ta.ADX(df.high, df.low, df.close, timeperiod=14)
    # ATR
    df['ATR_14'] = ta.ATR(df.high, df.low, df.close, timeperiod=14)
    # Bollinger BandWidth
    df['Bollinger_BandWidth_20'] = ta.STDDEV(df.close, timeperiod=20, nbdev=1) / df['BBANDS_20_MID']
    # Chaikin Money Flow
    df['Money_Flow_Volume'] = df.volume * (df.close - df.low - df.high + df.close) / (df.high - df.low)
    df['CMF_20'] = ta.MA(df['Money_Flow_Volume'], timeperiod=20) / ta.MA(df.volume, timeperiod=20)
    # Chaikin Oscillator
    df['Chaikin_Oscillator'] = ta.EMA(df['ADL'], timeperiod=3) - ta.EMA(df['ADL'], timeperiod=10)
    # CCI
    df['CCI_20'] = ta.CCI(df.high, df.low, df.close, timeperiod=20)
    df['CCI_40'] = ta.CCI(df.high, df.low, df.close, timeperiod=40)
    # Coppock Curve
    df['ROC_14'] = ta.ROC(df.close, timeperiod=14)
    df['ROC_11'] = ta.ROC(df.close, timeperiod=11)
    df['Coppock_Curve_10'] = ta.WMA(df['ROC_14'] + df['ROC_11'], timeperiod=10)
    # Detrended Price Oscillator (DPO)
    df['EMA_20'] = ta.EMA(df.close, timeperiod=20)
    df['DPO_11'] = df['EMA_20'].shift(11)
    # Ease of Movement
    df['EMV_1period'] = ((df.high + df.low) / 2 - (df.high.shift(1) + df.low.shift(1)) / 2) / (
                (df.volume / 100000000) / (df.high - df.low))
    df['EMV_14'] = ta.SMA(df['EMV_1period'], timeperiod=14)
    # Force Index
    df['Force_index_1'] = (df.close - df.close.shift(1)) * df.volume
    df['Force_index_13'] = ta.SMA(df['Force_index_1'], timeperiod=13)
    # Mass Index
    df['Single_EMA'] = ta.EMA(df.high - df.low, timeperiod=9)
    df['Double_EMA'] = ta.EMA(df['Single_EMA'], timeperiod=9)
    df['EMA_Ratio'] = df['Single_EMA'] / df['Double_EMA']
    df['Mass_Index_25'] = df['EMA_Ratio'].rolling(25).sum()
    # MACD
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = ta.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)
    # Money Flow Index (MFI)
    df['positive_flow'] = df.volume[df.close - df.close.shift(1) > 0]
    df['negative_flow'] = df.volume[df.close - df.close.shift(1) <= 0]
    df['positive_flow'].fillna(0, inplace=True)
    df['negative_flow'].fillna(0, inplace=True)
    df['Money_Flow_Ratio_14'] = ta.MA(df['positive_flow'], timeperiod=14) / ta.MA(df['negative_flow'], timeperiod=14)
    df['Money_Flow_Index_14'] = 100 - 100 / (1 + df['Money_Flow_Ratio_14'])
    # Negative Volume Index (NVI)
    df['close_pctchange'] = df.close.pct_change()
    df['volume_pctchange'] = df.volume.pct_change()
    df['NVI_value'] = df[df['volume_pctchange'] < 0]['close_pctchange']
    df['NVI_value'].fillna(0, inplace=True)
    df['NVI'] = df['NVI_value'].cumsum() + 1000
    df['NVI_signal'] = ta.EMA(df['NVI'], timeperiod=255)
    # OBV
    df['OBV'] = 0
    df.loc[df['close_pctchange'] > 0, 'OBV'] = df.volume
    df.loc[df['close_pctchange'] < 0, 'OBV'] = -df.volume
    df['OBV'] = df['OBV'].cumsum()
    df['OBV_signal_65'] = ta.SMA(df['OBV'], timeperiod=65)
    df['OBV_signal_20'] = ta.SMA(df['OBV'], timeperiod=20)
    # Percentage Price Oscillator
    df['PPO'] = ta.PPO(df.close, fastperiod=12, slowperiod=26, matype=0)
    df['PPO_signal'] = ta.EMA(df['PPO'], timeperiod=9)
    df['PPO_hist'] = df['PPO'] - df['PPO_signal']
    # Percentage Volume Oscillator
    df['PVO'] = ta.PPO(df.volume, fastperiod=12, slowperiod=26, matype=0)
    df['PVO_signal'] = ta.EMA(df['PVO'], timeperiod=9)
    df['PVO_hist'] = df['PVO'] - df['PVO_signal']
    #
    df['ROC_10'] = ta.ROC(df.close, timeperiod=10)
    df['ROC_15'] = ta.ROC(df.close, timeperiod=15)
    df['ROC_20'] = ta.ROC(df.close, timeperiod=20)
    df['ROC_30'] = ta.ROC(df.close, timeperiod=30)
    df['RCMA1'] = ta.SMA(df['ROC_10'], timeperiod=10)
    df['RCMA2'] = ta.SMA(df['ROC_15'], timeperiod=10)
    df['RCMA3'] = ta.SMA(df['ROC_20'], timeperiod=10)
    df['RCMA4'] = ta.SMA(df['ROC_30'], timeperiod=15)
    df['KST'] = df['RCMA1'] * 1 + df['RCMA2'] * 2 + df['RCMA3'] * 3 + df['RCMA4'] * 4
    df['KST_signal_9'] = ta.SMA(df['KST'], timeperiod=9)
    # Martin Pring's Special K
    df['ROC_40'] = ta.ROC(df.close, timeperiod=40)
    df['ROC_65'] = ta.ROC(df.close, timeperiod=65)
    df['ROC_75'] = ta.ROC(df.close, timeperiod=75)
    df['ROC_100'] = ta.ROC(df.close, timeperiod=100)
    df['RCMA5'] = ta.SMA(df['ROC_40'], timeperiod=40)
    df['RCMA6'] = ta.SMA(df['ROC_65'], timeperiod=65)
    df['RCMA7'] = ta.SMA(df['ROC_75'], timeperiod=75)
    df['RCMA8'] = ta.SMA(df['ROC_100'], timeperiod=100)
    df['SpecialK'] = df['KST'] + df['RCMA5'] * 1 + df['RCMA6'] * 2 + df['RCMA7'] * 3 + df['RCMA8'] * 4
    df['SpecialK_signal_100'] = ta.SMA(df['SpecialK'], timeperiod=100)
    # ROC
    df['ROC_125'] = ta.ROC(df.close, timeperiod=125)
    df['ROC_250'] = ta.ROC(df.close, timeperiod=250)
    # RSI
    df['RSI_5'] = ta.RSI(df.close, timeperiod=5)
    df['RSI_14'] = ta.RSI(df.close, timeperiod=14)
    # SLOPE
    df['LINEARREG_SLOPE_20'] = ta.LINEARREG_SLOPE(df.close, timeperiod=20)
    df['LINEARREG_SLOPE_100'] = ta.LINEARREG_SLOPE(df.close, timeperiod=100)
    # Standard Deviation (Volatility)
    df['STDDEV_10'] = ta.STDDEV(df.close, timeperiod=10, nbdev=1)
    df['STDDEV_250'] = ta.STDDEV(df.close, timeperiod=250, nbdev=1)
    # Stochastic Oscillator
    df['high_14'] = ta.MAX(df.high, 14)
    df['low_14'] = ta.MIN(df.low, 14)
    df['%K_14'] = 100 * (df.close - df['low_14']) / (df['high_14'] - df['low_14'])
    df['%D_3'] = ta.SMA(df['%K_14'], timeperiod=3)
    # StochRSI
    df['RIS_high_14'] = ta.MAX(df['RSI_14'], 14)
    df['RIS_low_14'] = ta.MIN(df['RSI_14'], 14)
    df['StochRSI_14'] = (df['RSI_14'] - df['RIS_low_14']) / (df['RIS_high_14'] - df['RIS_low_14'])
    # TRIX
    df['TRIX_15'] = ta.TRIX(df.close, timeperiod=15)
    df['TRIX_signal_9'] = ta.EMA(df['TRIX_15'], timeperiod=9)
    # True Strength Index (TSI)
    df['PC'] = df.close - df.close.shift(1)
    df['abs_PC'] = np.abs(df['PC'])
    df['First_Smoothing_PC_25'] = ta.EMA(df['PC'], timeperiod=25)
    df['Second_Smoothing_PC_13'] = ta.EMA(df['First_Smoothing_PC_25'], timeperiod=13)
    df['First_Smoothing_absPC_25'] = ta.EMA(df['abs_PC'], timeperiod=25)
    df['Second_Smoothing_absPC_13'] = ta.EMA(df['First_Smoothing_absPC_25'], timeperiod=13)
    df['TSI'] = 100 * df['Second_Smoothing_PC_13'] / df['Second_Smoothing_absPC_13']
    df['TSI_signal_7'] = ta.EMA(df['TSI'], timeperiod=7)
    # Ulcer Index
    df['max_close_14'] = ta.MAX(df.close, 14)
    df['Percent_Drawdown_14'] = 100 * (df.close - df['max_close_14']) / df['max_close_14']
    df['Ulcer_Index_14'] = ta.MA(df['Percent_Drawdown_14'], timeperiod=14)
    # Ultimate Oscillator
    df['BP'] = df.close.shift(1)
    df.loc[df['BP'] > df.low, 'BP'] = df.low
    df['TR'] = df.close.shift(1)
    df.loc[df['TR'] < df.high, 'TR'] = df.high
    df['TR'] = df['TR'] - df['BP']
    df['BP'] = df.close - df['BP']
    average7 = ta.MA(df['BP'], timeperiod=7) / ta.MA(df['TR'], timeperiod=7)
    average14 = ta.MA(df['BP'], timeperiod=14) / ta.MA(df['TR'], timeperiod=14)
    average28 = ta.MA(df['BP'], timeperiod=28) / ta.MA(df['TR'], timeperiod=28)
    df['UO'] = 100 * (4 * average7 + 2 * average14 + average28) / 7
    # Vortex Indicator
    df['Plus_VM'] = abs(df.high - df.low.shift(1))
    df['Minus_VM'] = abs(df.low - df.high.shift(1))
    df['Plus_VM_14'] = 14 * ta.MA(df['Plus_VM'], timeperiod=14)
    df['Minus_VM_14'] = 14 * ta.MA(df['Minus_VM'], timeperiod=14)
    df['H-L'] = df.high - df.low
    df['H-pC'] = abs(df.high - df.close.shift(1))
    df['L-pC'] = abs(df.low - df.close.shift(1))
    df['TR'] = df[['H-L', 'H-pC', 'L-pC']].max(axis=1)
    df['TR_14'] = 14 * ta.MA(df['TR'], timeperiod=14)
    df['Plus_VI_14'] = df['Plus_VM_14'] / df['TR_14']
    df['Minus_VI_14'] = df['Minus_VM_14'] / df['TR_14']
    # Williams %R
    df['WILLR_14'] = ta.WILLR(df.high, df.low, df.close, timeperiod=14)

def get_talib_indicators(df, freq_lst=(5, 15, 30), divided_by_close=False):
    """
    Additional indicators with different indicator parameters customized
    :param df: pandas.DataFrame with {'open', 'close', 'high', 'low', 'volume'} and index == date
    :return: pandas.DataFrame
    """
    for n in freq_lst:
        # Overlap Studies Functions
        #BBANDS
        df['BBANDS_' + str(n) + '_UP'], df['BBANDS_' + str(n) + '_MID'], df['BBANDS_' + str(n) + '_LOW'] = ta.BBANDS(df.close, timeperiod=n, nbdevup=2, nbdevdn=2, matype=0)
        #DEMA
        df['DEMA' + str(n)] = ta.DEMA(df.close, timeperiod=n)
        #EMA
        df['EMA' + str(n)] = ta.EMA(df.close, timeperiod=n)
        #KAMA
        df['KAMA' + str(n)] = ta.KAMA(df.close, timeperiod=n)
        #MA
        df['MA' + str(n)] = ta.MA(df.close, timeperiod=n)
        #MAMA(ERROR)
        #df['MAMA' + str(n)] = ta.MAMA(df.close, fastlimit=0, slowlimit=0)
        #MAVP(periods 参数暂时无法确定)
        #df['' + str(n)] = ta.MAVP(df.close, periods, minperiod=2, maxperiod=30, matype=0)
        #MIDPOINT
        df['MIDPOINT' + str(n)] = ta.MIDPOINT(df.close, timeperiod=n)
        #MIDPRICE
        df['MIDPRICE' + str(n)] = ta.MIDPRICE(df.high, df.low, timeperiod=n)
        #SAR
        df['SAR' + str(n)] = ta.SAR(df.high, df.low, acceleration=0, maximum=0)
        #SAREXT（参数太多，暂时无法知道如何设置）
        #df['SAREXT' + str(n)] = ta.SAREXT(df.high, df.low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
        #SMA
        df['SMA' + str(n)] = ta.SMA(df.close, timeperiod=n)
        #T3
        df['T3' + str(n)] = ta.T3(df.close, timeperiod=n, vfactor=0.7)
        #TEMA
        df['TEMA' + str(n)] = ta.TEMA(df.close, timeperiod=n)
        #TRIMA
        df['TRIMA' + str(n)] = ta.TRIMA(df.close, timeperiod=n)
        #WMA
        df['WMA' + str(n)] = ta.WMA(df.close, timeperiod=n)
        #Momentum Indicator Functions
        #ADX
        df['ADX' + str(n)] = ta.ADX(df.high, df.low, df.close, timeperiod=n)
        #ADXR
        df['ADXR' + str(n)] = ta.ADXR(df.high, df.low, df.close, timeperiod=n)
        #APO
        df['APO' + str(n)] = ta.APO(df.close, fastperiod=12, slowperiod=26, matype=0)
        #AROON
        df['AROON_DOWN_' + str(n)], df['AROON_UP_' + str(n)] = ta.AROON(df.high, df.low, timeperiod=n)
        #AROONOSC
        df['AROONOSC' + str(n)] = ta.AROONOSC(df.high, df.low, timeperiod=n)
        #CCI
        df['CCI' + str(n)] = ta.CCI(df.high, df.low, df.close, timeperiod=n)
        #CMO
        df['CMO' + str(n)] = ta.CMO(df.close, timeperiod=n)
        #DX
        df['DX' + str(n)] = ta.DX(df.high, df.low, df.close, timeperiod=n)
        #MFI
        df['MFI' + str(n)] = ta.MFI(df.high, df.low, df.close, df.volume, timeperiod=n)
        #MINUS_DI
        df['MINUS_DI' + str(n)] = ta.MINUS_DI(df.high, df.low, df.close, timeperiod=n)
        #MINUS_DM
        df['MINUS_DM' + str(n)] = ta.MINUS_DM(df.high, df.low, timeperiod=n)
        #MOM
        df['MOM' + str(n)] = ta.MOM(df.close, timeperiod=n)
        #PLUS_DI
        df['PLUS_DI' + str(n)] = ta.PLUS_DI(df.high, df.low, df.close, timeperiod=n)
        #PLUS_DM
        df['PLUS_DM' + str(n)] = ta.PLUS_DM(df.high, df.low, timeperiod=n)
        #ROC
        df['ROC' + str(n)] = ta.ROC(df.close, timeperiod=n)
        #ROCP
        df['ROCP' + str(n)] = ta.ROCP(df.close, timeperiod=n)
        #ROCR
        df['ROCR' + str(n)] = ta.ROCR(df.close, timeperiod=n)
        #ROCR100
        df['ROCR100_' + str(n)] = ta.ROCR100(df.close, timeperiod=n)
        #RSI
        df['RSI' + str(n)] = ta.RSI(df.close, timeperiod=n)
        #TRIX
        df['TRIX' + str(n)] = ta.TRIX(df.close, timeperiod=n)
        #WILLR
        df['WILLR' + str(n)] = ta.WILLR(df.high, df.low, df.close, timeperiod=n)

        #Volatility Indicator Functions
        #ATR
        df['ATR' + str(n)] = ta.ATR(df.high, df.low, df.close, timeperiod=n)
        #NATR
        df['NATR' + str(n)] = ta.NATR(df.high, df.low, df.close, timeperiod=n)

        # Statistic Functions
        #BETA
        df['BETA' + str(n)] = ta.BETA(df.high, df.low, timeperiod=n)
        #CORREL
        df['CORREL' + str(n)] = ta.CORREL(df.high, df.low, timeperiod=n)
        #LINEARREG
        df['LINEARREG' + str(n)] = ta.LINEARREG(df.close, timeperiod=n)
        #LINEARREG_ANGLE
        df['LINEARREG_ANGLE' + str(n)] = ta.LINEARREG_ANGLE(df.close, timeperiod=n)
        #LINEARREG_INTERCEPT
        df['LINEARREG_INTERCEPT' + str(n)] = ta.LINEARREG_INTERCEPT(df.close, timeperiod=n)
        #LINEARREG_SLOPE
        df['LINEARREG_SLOPE' + str(n)] = ta.LINEARREG_SLOPE(df.close, timeperiod=n)
        #STDDEV
        df['STDDEV' + str(n)] = ta.STDDEV(df.close, timeperiod=n, nbdev=1)
        #TSF
        df['TSF' + str(n)] = ta.TSF(df.close, timeperiod=n)
        #VAR
        df['VAR' + str(n)] = ta.VAR(df.close, timeperiod=n, nbdev=1)
        # Math Operator Functions
        #MAX
        df['MAX' + str(n)] = ta.MAX(df.close, timeperiod=n)
        #MAXINDEX
        #df['MAXINDEX' + str(n)] = ta.MAXINDEX(df.close, timeperiod=n)
        #MIN
        df['MIN' + str(n)] = ta.MIN(df.close, timeperiod=n)
        #MININDEX
        #df['MININDEX' + str(n)] = ta.MININDEX(df.close, timeperiod=n)
        #SUM
        #df['SUM' + str(n)] = ta.SUM(df.close, timeperiod=n)

    # HT_TRENDLINE
    df['HT_TRENDLINE'] = ta.HT_TRENDLINE(df.close)
    # BOP
    df['BOP'] = ta.BOP(df.open, df.high, df.low, df.close)
    # MACD
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = ta.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)
    # STOCH
    df['slowk'], df['slowd'] = ta.STOCH(df.high, df.low, df.close, fastk_period=5, slowk_period=3,
                                                          slowk_matype=0, slowd_period=3, slowd_matype=0)
    # STOCHF
    df['fsatk'], df['fastd'] = ta.STOCHF(df.high, df.low, df.close, fastk_period=5, fastd_period=3,
                                                           fastd_matype=0)
    # STOCHRSI
    df['fsatk_RSI'], df['fastd_RSI'] = ta.STOCHRSI(df.close, timeperiod=n, fastk_period=5,
                                                                     fastd_period=3, fastd_matype=0)
    # PPO
    df['PPO'] = ta.PPO(df.close, fastperiod=12, slowperiod=26, matype=0)
    # ULTOSC
    df['ULTOSC'] = ta.ULTOSC(df.high, df.low, df.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    # Volume Indicator Functions
    # AD
    #df['AD'] = ta.AD(df.high, df.low, df.close, df.volume)
    # ADOSC
    #df['ADOSC'] = ta.ADOSC(df.high, df.low, df.close, df.volume, fastperiod=3, slowperiod=10)
    # OBV
    # OBV is too large
    #df['OBV'] = ta.OBV(df.close, df.volume)
    # TRANGE
    df['TRANGE'] = ta.TRANGE(df.high, df.low, df.close)
    # Price Indicator Functions
    # AVGPRICE
    df['AVGPRICE'] = ta.AVGPRICE(df.open, df.high, df.low, df.close)
    # MEDPRICE
    df['MEDPRICE'] = ta.MEDPRICE(df.high, df.low)
    # TYPPRICE
    df['TYPPRICE'] = ta.TYPPRICE(df.high, df.low, df.close)
    # WCLPRICE
    df['WCLPRICE'] = ta.WCLPRICE(df.high, df.low, df.close)
    # Cycle Indicator Functions
    # HT_DCPERIOD
    df['HT_DCPERIOD'] = ta.HT_DCPERIOD(df.close)
    # HT_DCPHASE
    df['HT_DCPHASE'] = ta.HT_DCPHASE(df.close)
    # HT_PHASOR
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = ta.HT_PHASOR(df.close)
    # HT_SINE
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = ta.HT_SINE(df.close)
    # HT_TRENDMODE
    df['HT_TRENDMODE'] = ta.HT_TRENDMODE(df.close)

    #Indicators regardless of time_period
    #Pattern Recognition Functions
    #CDL2CROWS
    '''df['CDL2CROWS'] = ta.CDL2CROWS(df.open, df.high, df.low, df.close)
    # CDL3BLACKCROWS
    df['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(df.open, df.high, df.low, df.close)
    # CDL3INSIDE
    df['CDL3INSIDE'] = ta.CDL3INSIDE(df.open, df.high, df.low, df.close)
    # CDL3LINESTRIKE
    df['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(df.open, df.high, df.low, df.close)
    # CDL3OUTSIDE
    df['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(df.open, df.high, df.low, df.close)
    # CDL3STARSINSOUTH
    df['CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(df.open, df.high, df.low, df.close)
    # CDL3WHITESOLDIERS
    df['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df.open, df.high, df.low, df.close)
    # CDLABANDONEDBABY
    df['CDLABANDONEDBABY'] = ta.CDLABANDONEDBABY(df.open, df.high, df.low, df.close, penetration=0)
    # CDLADVANCEBLOCK
    df['CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(df.open, df.high, df.low, df.close)
    # CDLBELTHOLD
    df['CDLBELTHOLD'] = ta.CDLBELTHOLD(df.open, df.high, df.low, df.close)
    # CDLBREAKAWAY
    df['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(df.open, df.high, df.low, df.close)
    # CDLCLOSINGMARUBOZU
    df['CDLCLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(df.open, df.high, df.low, df.close)
    # CDLCONCEALBABYSWALL
    df['CDLCONCEALBABYSWALL'] = ta.CDLCONCEALBABYSWALL(df.open, df.high, df.low, df.close)
    # CDLCOUNTERATTACK
    df['CDLCOUNTERATTACK'] = ta.CDLCOUNTERATTACK(df.open, df.high, df.low, df.close)
    # CDLDARKCLOUDCOVER
    df['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df.open, df.high, df.low, df.close, penetration=0)
    # CDLDOJI
    df['CDLDOJI'] = ta.CDLDOJI(df.open, df.high, df.low, df.close)
    # CDLDOJISTAR
    df['CDLDOJISTAR'] = ta.CDLDOJISTAR(df.open, df.high, df.low, df.close)
    # CDLDRAGONFLYDOJI
    df['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(df.open, df.high, df.low, df.close)
    # CDLENGULFING
    df['CDLENGULFING'] = ta.CDLENGULFING(df.open, df.high, df.low, df.close)
    # CDLEVENINGDOJISTAR
    df['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(df.open, df.high, df.low, df.close, penetration=0)
    # CDLEVENINGSTAR
    df['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df.open, df.high, df.low, df.close, penetration=0)
    # CDLGAPSIDESIDEWHITE
    df['CDLGAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(df.open, df.high, df.low, df.close)
    # CDLGRAVESTONEDOJI
    df['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(df.open, df.high, df.low, df.close)
    # CDLHAMMER
    df['CDLHAMMER'] = ta.CDLHAMMER(df.open, df.high, df.low, df.close)
    # CDLHANGINGMAN
    df['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df.open, df.high, df.low, df.close)
    # CDLHARAMI
    df['CDLHARAMI'] = ta.CDLHARAMI(df.open, df.high, df.low, df.close)
    # CDLHARAMICROSS
    df['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(df.open, df.high, df.low, df.close)
    # CDLHIGHWAVE
    df['CDLHIGHWAVE'] = ta.CDLHIGHWAVE(df.open, df.high, df.low, df.close)
    # CDLHIKKAKE
    df['CDLHIKKAKE'] = ta.CDLHIKKAKE(df.open, df.high, df.low, df.close)
    # CDLHIKKAKEMOD
    df['CDLHIKKAKEMOD'] = ta.CDLHIKKAKEMOD(df.open, df.high, df.low, df.close)
    # CDLHOMINGPIGEON
    df['CDLHOMINGPIGEON'] = ta.CDLHOMINGPIGEON(df.open, df.high, df.low, df.close)
    # CDLIDENTICAL3CROWS
    df['CDLIDENTICAL3CROWS'] = ta.CDLIDENTICAL3CROWS(df.open, df.high, df.low, df.close)
    # CDLINNECK
    df['CDLINNECK'] = ta.CDLINNECK(df.open, df.high, df.low, df.close)
    # CDLINVERTEDHAMMER
    df['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df.open, df.high, df.low, df.close)
    # CDLKICKING
    df['CDLKICKING'] = ta.CDLKICKING(df.open, df.high, df.low, df.close)
    # CDLKICKINGBYLENGTH
    df['CDLKICKINGBYLENGTH'] = ta.CDLKICKINGBYLENGTH(df.open, df.high, df.low, df.close)
    # CDLLADDERBOTTOM
    df['CDLLADDERBOTTOM'] = ta.CDLLADDERBOTTOM(df.open, df.high, df.low, df.close)
    # CDLLONGLEGGEDDOJI
    df['CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(df.open, df.high, df.low, df.close)
    # CDLLONGLINE
    df['CDLLONGLINE'] = ta.CDLLONGLINE(df.open, df.high, df.low, df.close)
    # CDLMARUBOZU
    df['CDLMARUBOZU'] = ta.CDLMARUBOZU(df.open, df.high, df.low, df.close)
    # CDLMATCHINGLOW
    df['CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(df.open, df.high, df.low, df.close)
    # CDLMATHOLD
    df['CDLMATHOLD'] = ta.CDLMATHOLD(df.open, df.high, df.low, df.close, penetration=0)
    # CDLMORNINGDOJISTAR
    df['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(df.open, df.high, df.low, df.close, penetration=0)
    # CDLMORNINGSTAR
    df['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df.open, df.high, df.low, df.close, penetration=0)
    # CDLONNECK
    df['CDLONNECK'] = ta.CDLONNECK(df.open, df.high, df.low, df.close)
    # CDLPIERCING
    df['CDLPIERCING'] = ta.CDLPIERCING(df.open, df.high, df.low, df.close)
    # CDLRICKSHAWMAN
    df['CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(df.open, df.high, df.low, df.close)
    # CDLRISEFALL3METHODS
    df['CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(df.open, df.high, df.low, df.close)
    # CDLSEPARATINGLINES
    df['CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(df.open, df.high, df.low, df.close)
    # CDLSHOOTINGSTAR
    df['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df.open, df.high, df.low, df.close)
    # CDLSHORTLINE
    df['CDLSHORTLINE'] = ta.CDLSHORTLINE(df.open, df.high, df.low, df.close)
    # CDLSPINNINGTOP
    df['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df.open, df.high, df.low, df.close)
    # CDLSTALLEDPATTERN
    df['CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(df.open, df.high, df.low, df.close)
    # CDLSTICKSANDWICH
    df['CDLSTICKSANDWICH'] = ta.CDLSTICKSANDWICH(df.open, df.high, df.low, df.close)
    # CDLTAKURI
    df['CDLTAKURI'] = ta.CDLTAKURI(df.open, df.high, df.low, df.close)
    # CDLTASUKIGAP
    df['CDLTASUKIGAP'] = ta.CDLTASUKIGAP(df.open, df.high, df.low, df.close)
    # CDLTHRUSTING
    df['CDLTHRUSTING'] = ta.CDLTHRUSTING(df.open, df.high, df.low, df.close)
    # CDLTRISTAR
    df['CDLTRISTAR'] = ta.CDLTRISTAR(df.open, df.high, df.low, df.close)
    # CDLUNIQUE3RIVER
    df['CDLUNIQUE3RIVER'] = ta.CDLUNIQUE3RIVER(df.open, df.high, df.low, df.close)
    # CDLUPSIDEGAP2CROWS
    df['CDLUPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(df.open, df.high, df.low, df.close)
    # CDLXSIDEGAP3METHODS
    df['CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(df.open, df.high, df.low, df.close)'''
    # Math Transform Functions
    #ACOS
    #df['ACOS'] = ta.ACOS(df.close)
    #ASIN
    #df['ASIN'] = ta.ASIN(df.close)
    #ATAN
    df['ATAN'] = ta.ATAN(df.close)
    #CEIL
    #df['CEIL'] = ta.CEIL(df.close)
    #COS
    df['COS'] = ta.COS(df.close)
    #COSH
    #df['COSH'] = ta.COSH(df.close)
    #EXP
    #df['EXP'] = ta.EXP(df.close)
    #FLOOR
    #df['FLOOR'] = ta.FLOOR(df.close)
    #LN
    df['LN'] = ta.LN(df.close)
    #LOG10
    df['LOG10'] = ta.LOG10(df.close)
    #SIN
    df['SIN'] = ta.SIN(df.close)
    #SINH
    #df['SINH'] = ta.SINH(df.close)
    #SQRT
    df['SQRT'] = ta.SQRT(df.close)
    #TAN
    df['TAN'] = ta.TAN(df.close)
    #TANH
    df['TANH'] = ta.TANH(df.close)
    # Math Operator Functions
    # ADD
    #df['ADD'] = ta.ADD(df.high, df.low)
    # DIV
    #df['DIV'] = ta.DIV(df.high, df.low)
    # MULT
    #df['MULT'] = ta.MULT(df.high, df.low)
    # ADD
    #df['SUB'] = ta.SUB(df.high, df.low)

    if divided_by_close:
        bias = 1
        df['BBANDS_5_UP'] = df['BBANDS_5_UP'] / df.close - bias
        df['BBANDS_5_MID'] = df['BBANDS_5_MID'] / df.close - bias
        df['BBANDS_5_LOW'] = df['BBANDS_5_LOW'] / df.close - bias
        df['DEMA5'] = df['DEMA5'] / df.close - bias
        df['EMA5'] = df['EMA5'] / df.close - bias
        df['KAMA5'] = df['KAMA5'] / df.close - bias
        df['MA5'] = df['MA5'] / df.close - bias
        df['MIDPOINT5'] = df['MIDPOINT5'] / df.close - bias
        df['MIDPRICE5'] = df['MIDPRICE5'] / df.close - bias
        df['SMA5'] = df['SMA5'] / df.close - bias
        df['T35'] = df['T35'] / df.close - bias
        df['TEMA5'] = df['TEMA5'] / df.close - bias
        df['TRIMA5'] = df['TRIMA5'] / df.close - bias
        df['WMA5'] = df['WMA5'] / df.close - bias
        df['LINEARREG5'] = df['LINEARREG5'] / df.close - bias
        df['LINEARREG_INTERCEPT5'] = df['LINEARREG_INTERCEPT5'] / df.close - bias
        df['TSF5'] = df['TSF5'] / df.close - bias
        df['MAX5'] = df['MAX5'] / df.close - bias
        df['MIN5'] = df['MIN5'] / df.close - bias
        df['BBANDS_15_UP'] = df['BBANDS_15_UP'] / df.close - bias
        df['BBANDS_15_MID'] = df['BBANDS_15_MID'] / df.close - bias
        df['BBANDS_15_LOW'] = df['BBANDS_15_LOW'] / df.close - bias
        df['DEMA15'] = df['DEMA15'] / df.close - bias
        df['EMA15'] = df['EMA15'] / df.close - bias
        df['KAMA15'] = df['KAMA15'] / df.close - bias
        df['MA15'] = df['MA15'] / df.close - bias
        df['MIDPOINT15'] = df['MIDPOINT15'] / df.close - bias
        df['MIDPRICE15'] = df['MIDPRICE15'] / df.close - bias
        df['SMA15'] = df['SMA15'] / df.close - bias
        df['T315'] = df['T315'] / df.close - bias
        df['TEMA15'] = df['TEMA15'] / df.close - bias
        df['TRIMA15'] = df['TRIMA15'] / df.close - bias
        df['WMA15'] = df['WMA15'] / df.close - bias
        df['LINEARREG15'] = df['LINEARREG15'] / df.close - bias
        df['LINEARREG_INTERCEPT15'] = df['LINEARREG_INTERCEPT15'] / df.close - bias
        df['TSF15'] = df['TSF15'] / df.close - bias
        df['MAX15'] = df['MAX15'] / df.close - bias
        df['MIN15'] = df['MIN15'] / df.close - bias
        df['BBANDS_30_UP'] = df['BBANDS_30_UP'] / df.close - bias
        df['BBANDS_30_MID'] = df['BBANDS_30_MID'] / df.close - bias
        df['BBANDS_30_LOW'] = df['BBANDS_30_LOW'] / df.close - bias
        df['DEMA30'] = df['DEMA30'] / df.close - bias
        df['EMA30'] = df['EMA30'] / df.close - bias
        df['KAMA30'] = df['KAMA30'] / df.close - bias
        df['MA30'] = df['MA30'] / df.close - bias
        df['MIDPOINT30'] = df['MIDPOINT30'] / df.close - bias
        df['MIDPRICE30'] = df['MIDPRICE30'] / df.close - bias
        df['SMA30'] = df['SMA30'] / df.close - bias
        df['T330'] = df['T330'] / df.close - bias
        df['TEMA30'] = df['TEMA30'] / df.close - bias
        df['TRIMA30'] = df['TRIMA30'] / df.close - bias
        df['WMA30'] = df['WMA30'] / df.close - bias
        df['LINEARREG_INTERCEPT30'] = df['LINEARREG_INTERCEPT30'] / df.close - bias
        df['TSF30'] = df['TSF30'] / df.close - bias
        df['MAX30'] = df['MAX30'] / df.close - bias
        df['MIN30'] = df['MIN30'] / df.close - bias
        df['HT_TRENDLINE'] = df['HT_TRENDLINE'] / df.close - bias
        df['AVGPRICE'] = df['AVGPRICE'] / df.close - bias
        df['MEDPRICE'] = df['MEDPRICE'] / df.close - bias
        df['TYPPRICE'] = df['TYPPRICE'] / df.close - bias
        df['WCLPRICE'] = df['WCLPRICE'] / df.close - bias
        df['open'] = df['open'] / df.close - bias
        df['high'] = df['high'] / df.close - bias
        df['low'] = df['low'] / df.close - bias
        #df['close'] = df['close'] / df.close - bias


def get_last_data(df):
    for i in [1,2,3,5,10,15,30,60,180,252]:
        df['last_price_{}'.format(i)] = df.close.shift(i)
        df['last_volume_{}'.format(i)] = df.volume.shift(i)




if __name__ == '__main__':
    df = pd.read_csv(zsys.rdatCN + '600016.csv')
    df = get_all_finanical_indicators(df)
    print(df.head())

