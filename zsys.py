# -*- coding: utf-8 -*- 
"""
The module "zsys" includes basic file path, indicator compositions
"""
import os


# rdat0='/WORK/quant/TQDat/'

rdat0 = os.path.abspath('.') + '/data/'
rdatCN0=rdat0+"day/"
rdatCN=rdatCN0+"stk/"
rdatCNX=rdatCN0+"inx/"
rdatCNE=rdatCN0+"etf/"
rdatInx=rdat0+"data/"
rdatMin0=rdat0+"min/"
rdatTick=rdat0+"tick/"
rdatReal=rdat0+"real/"

# Mysql info
use_mysql = True
mysql_user = 'root'
mysql_password = 'admin'
mysql_host = '192.168.1.4'
mysql_port = '3309'


#
#
#ohlc=['open','high','low','close']
#ohlc_date=['date']+ohlc
#
#---qxLib.xxxx
ohlcLst=['open','high','low','close']
ohlcVLst=ohlcLst+['volume']
ohlcVALst=ohlcLst+['volume','avg']
ohlcALst=ohlcLst+['avg']
ohlVLst=['open','high','low','volume']
#
ohlcDLst=['date']+ohlcLst
ohlcDVLst=['date']+ohlcLst+['volume']
ohlcExtLst=ohlcDLst+['volume','adj close']
#
xavg9Lst=['xavg1','xavg2','xavg3','xavg4','xavg5','xavg6','xavg7','xavg8','xavg9']
xavg5Lst=['xavg1','xavg2','xavg3','xavg4','xavg5']
# 
ma100Lst_var=[2,3,5,10,15,20,25,30,50,100]
ma100Lst=['ma_2','ma_3','ma_5','ma_10','ma_15','ma_20','ma_25','ma_30','ma_50','ma_100']
ma200Lst_var=[2,3,5,10,15,20,25,30,50,100,150,200]
#ma200Lst=['ma_2', 'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20','ma_25', 'ma_30', 'ma_50', 'ma_100', 'ma_150', 'ma_200']
ma200Lst=['ma_2', 'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20','ma_30', 'ma_50', 'ma_100', 'ma_150', 'ma_200']
#
ma030Lst_var=[2,3,5,10,15,20,25,30]
ma030Lst=['ma_2', 'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20', 'ma_25', 'ma_30']
#
priceLst=['price', 'price_next', 'price_change']
#
dateLst=['xyear','xmonth','xday','xday_week','xday_year','xweek_year']
timeLst=dateLst+['xhour','xminute']
#
TDS_xlst1=ohlcVALst
TDS_xlst2=ohlcVALst+ma100Lst
TDS_xlst9=TDS_xlst2+dateLst
#
#  keras.fun-lst.xxx
k_init_lst=['glorot_uniform','random_uniform','Zeros','Ones','Ones','RandomNormal','RandomUniform','TruncatedNormal','VarianceScaling','Orthogonal','Identiy','lecun_uniform','lecun_normal','glorot_normal','glorot_uniform','he_normal','he_uniform']
f_act_lst=[None,'elu','selu','relu','tanh','linear','sigmoid','softplus','hard_sigmoid'] #,'softsign,'
f_out_typ_lst=[None,'softmax']
#
f_opt_lst=['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
f_loss_lst=[ 'mse','mae','mape','msle','squared_hinge','hinge','binary_crossentropy','logcosh','kullback_leibler_divergence','poisson','cosine_proximity','categorical_hinge']
f_loss_typ_lst=['categorical_crossentropy'] #softmax
#



#DJQ自定义变量
TDS_xlst_djq = ['open', 'high', 'low', 'close', 'volume', 'xday_week', 'xday_year', 'xweek_year', 'ma_2',
       'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20', 'ma_25', 'ma_30',
       'ma_50', 'ma_100', 'adx_5_14',
       'adx_10_14', 'adx_30_14', 'atr_5', 'atr_10', 'atr_30', 'boll_5b',
       'boll_5', 'boll_10b', 'boll_10', 'boll_30b', 'boll_30',
       'boll_ma_5', 'boll_std_5', 'boll_up_5', 'boll_low_5', 'boll_ma_10',
       'boll_std_10', 'boll_up_10', 'boll_low_10', 'boll_ma_30',
       'boll_std_30', 'boll_up_30', 'boll_low_30', 'cci_5', 'cci_10',
       'cci_30', 'copp_5', 'copp_10', 'copp_30', 'ck_5', 'ck_10', 'ck_30',
       'donch_5_sr', 'donch_5', 'donch_10_sr', 'donch_10', 'donch_30_sr',
       'donch_30', 'donch_5_up', 'donch_5_low', 'donch_5_mid',
       'donch_10_up', 'donch_10_low', 'donch_10_mid', 'donch_30_up',
       'donch_30_low', 'donch_30_mid', 'ema_5', 'ema_10', 'ema_30',
       'eom_5', 'eom_10', 'eom_30', 'fib', 'fib618', 'fib381', 'fib-381',
       'fib-618', 'force_5', 'force_10', 'force_30', 'rsv_5', 'kdj_k_5',
       'kdj_d_5', 'kdj_j_5', 'rsv_10', 'kdj_k_10', 'kdj_d_10', 'kdj_j_10',
       'rsv_30', 'kdj_k_30', 'kdj_d_30', 'kdj_j_30', 'macd', 'msign',
       'mdiff', 'macd_2', 'mdea_2', 'mdiff_2', 'mfi_5', 'mfi_10',
       'mfi_30', 'mom_5', 'mom_10', 'mom_30', 'mass', 'obv_5', 'obv_10',
       'obv_30', 'pp', 'r1', 's1', 'r2', 's2', 'r3', 's3', 'roc_5',
       'roc_10', 'roc_30', 'rsi_5', 'rsi_10', 'rsi_30', 'rsi100_5_k',
       'rsi100_5', 'rsi100_10_k', 'rsi100_10', 'rsi100_30_k', 'rsi100_30',
       'std_5', 'std_10', 'std_30', 'stok_5', 'stod_5', 'stok_10',
       'stod_10', 'stok_30', 'stod_30', 'trix_5', 'trix_10', 'trix_30',
       'tsi', 'uos', 'vortex_5', 'vortex_10', 'vortex_30']

TDS_talib_Overlap_5 = ['BBANDS_5_UP', 'BBANDS_5_MID',
                     'BBANDS_5_LOW', 'DEMA5', 'EMA5', 'KAMA5', 'MA5', 'MIDPOINT5', 'MIDPRICE5',
                     'SAR5', 'SMA5', 'T35', 'TEMA5', 'TRIMA5', 'WMA5']
TDS_talib_Momentum_5 = ['ADX5', 'ADXR5', 'APO5',
                     'AROON_DOWN_5', 'AROON_UP_5', 'AROONOSC5', 'CCI5', 'CMO5', 'DX5', 'MFI5',
                     'MINUS_DI5', 'MINUS_DM5', 'MOM5', 'PLUS_DI5', 'PLUS_DM5', 'ROC5', 'ROCP5',
                     'ROCR5', 'ROCR100_5', 'RSI5', 'TRIX5', 'WILLR5']
TDS_talib_Volatility_5 = ['ATR5', 'NATR5']
TDS_talib_Statistic_5 = ['BETA5','CORREL5', 'LINEARREG5', 'LINEARREG_ANGLE5', 'LINEARREG_INTERCEPT5', 'LINEARREG_SLOPE5',
                     'STDDEV5', 'TSF5', 'VAR5']
TDS_talib_Math_5 = ['MAX5', 'MIN5']
TDS_talib_Overlap_15 = ['BBANDS_15_UP', 'BBANDS_15_MID',
                     'BBANDS_15_LOW', 'DEMA15', 'EMA15', 'KAMA15', 'MA15', 'MIDPOINT15', 'MIDPRICE15',
                     'SAR15', 'SMA15', 'T315', 'TEMA15', 'TRIMA15', 'WMA15']
TDS_talib_Momentum_15 = ['ADX15', 'ADXR15',
                     'APO15', 'AROON_DOWN_15', 'AROON_UP_15', 'AROONOSC15', 'CCI15', 'CMO15',
                     'DX15', 'MFI15', 'MINUS_DI15', 'MINUS_DM15', 'MOM15', 'PLUS_DI15', 'PLUS_DM15',
                     'ROC15', 'ROCP15', 'ROCR15', 'ROCR100_15', 'RSI15', 'TRIX15', 'WILLR15']
TDS_talib_Volatility_15 = ['ATR15', 'NATR15']
TDS_talib_Statistic_15 = ['BETA15', 'CORREL15', 'LINEARREG15', 'LINEARREG_ANGLE15', 'LINEARREG_INTERCEPT15',
                     'LINEARREG_SLOPE15', 'STDDEV15', 'TSF15', 'VAR15']
TDS_talib_Math_15 = ['MAX15', 'MIN15']
TDS_talib_Overlap_30 = ['BBANDS_30_UP',
                     'BBANDS_30_MID', 'BBANDS_30_LOW', 'DEMA30', 'EMA30', 'KAMA30', 'MA30', 'MIDPOINT30',
                     'MIDPRICE30', 'SAR30', 'SMA30', 'T330', 'TEMA30', 'TRIMA30', 'WMA30']
TDS_talib_Momentum_30 = ['ADX30',
                     'ADXR30', 'APO30', 'AROON_DOWN_30', 'AROON_UP_30', 'AROONOSC30', 'CCI30', 'CMO30',
                     'DX30', 'MFI30', 'MINUS_DI30', 'MINUS_DM30', 'MOM30', 'PLUS_DI30', 'PLUS_DM30',
                     'ROC30', 'ROCP30', 'ROCR30', 'ROCR100_30', 'RSI30', 'TRIX30', 'WILLR30']
TDS_talib_Volatility_30 = ['ATR30', 'NATR30']
TDS_talib_Statistic_30 = ['BETA30', 'CORREL30', 'LINEARREG30', 'LINEARREG_ANGLE30', 'LINEARREG_INTERCEPT30',
                     'LINEARREG_SLOPE30', 'STDDEV30', 'TSF30', 'VAR30']
TDS_talib_Math_30 = ['MAX30', 'MIN30']
TDS_talib_Overlap_Static = ['HT_TRENDLINE']
TDS_talib_Momentum_Static = ['BOP', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'slowk', 'slowd', 'fsatk', 'fastd',
                     'fsatk_RSI', 'fastd_RSI', 'PPO', 'ULTOSC']
TDS_talib_Volume_Static = ['AD', 'ADOSC', 'OBV']
TDS_talib_Volatility_Static = ['TRANGE']
TDS_talib_Price_Static = ['AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE']
TDS_talib_Cycle_Static = ['HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR_inphase',
                     'HT_PHASOR_quadrature', 'HT_SINE_sine', 'HT_SINE_leadsine', 'HT_TRENDMODE']
TDS_talib_Pattern_Static = ['CDL2CROWS',
                     'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH',
                     'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY',
                     'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER',
                     'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
                     'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
                     'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON',
                     'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
                     'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW',
                     'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING',
                     'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR',
                     'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI',
                     'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
                     'CDLXSIDEGAP3METHODS']
TDS_talib_Math_Static = ['ATAN', 'COS', 'LN', 'LOG10', 'SIN',
                     'SQRT', 'TAN', 'TANH']
TDS_talib_multi_indicators_5 = TDS_talib_Overlap_5 + TDS_talib_Momentum_5 + TDS_talib_Volatility_5 + TDS_talib_Statistic_5 + TDS_talib_Math_5
TDS_talib_multi_indicators_15 = TDS_talib_Overlap_15 + TDS_talib_Momentum_15 + TDS_talib_Volatility_15 + TDS_talib_Statistic_15 + TDS_talib_Math_15
TDS_talib_multi_indicators_30 = TDS_talib_Overlap_30 + TDS_talib_Momentum_30 + TDS_talib_Volatility_30 + TDS_talib_Statistic_30 + TDS_talib_Math_30
TDS_talib_multi_indicators_Static = TDS_talib_Overlap_Static \
                                    + TDS_talib_Momentum_Static \
                                    + TDS_talib_Volatility_Static \
                                    + TDS_talib_Price_Static \
                                    + TDS_talib_Cycle_Static \
                                    #+ TDS_talib_Volume_Static \
                                    #+ TDS_talib_Math_Static \
                                    #+ TDS_talib_Pattern_Static
TDS_talib_indicators_all = TDS_talib_multi_indicators_5 + TDS_talib_multi_indicators_15 + TDS_talib_multi_indicators_30 + TDS_talib_multi_indicators_Static

stcokcharts_indicators = ['BBANDS_20_UP', 'BBANDS_20_MID', 'BBANDS_20_LOW', '%B_20', 'Chandelier_Exit_22_long',
                          'Chandelier_Exit_22_short', 'Conversion_Line_9', 'Base_Line_26', 'Leading_Span_A',
                          'Leading_Span_B', 'Lagging_Span', 'KAMA_10', 'Keltner_Channels_middle',
                          'Keltner_Channels_upper', 'Keltner_Channels_lower', 'SMA_20', 'SMA_50', 'SMA_60', 'SMA_150',
                          'EMA_60', 'SMA_volume_200', 'Upper_Envelope_20', 'Lower_Envelope_20', 'Upper_Envelope_50',
                          'Lower_Envelope_50', 'SAR_0.01_0.2', 'Pivot_Point', 'Support_1', 'Support_2', 'Support_3',
                          'Resistance_1', 'Resistance_2', 'Resistance_3', 'high_20', 'low_20', 'Price_Channels_center',
                          'VWAP', 'ADL', 'AROON_DOWN_25', 'AROON_UP_25',
                          'Aroon_Oscillator_25', 'MINUS_DI_14', 'MINUS_DM_14', 'PLUS_DI_14', 'PLUS_DM_14', 'ADX_14',
                          'ATR_14', 'Bollinger_BandWidth_20', 'Money_Flow_Volume', 'CMF_20', 'Chaikin_Oscillator',
                          'CCI_20', 'CCI_40', 'ROC_14', 'ROC_11', 'Coppock_Curve_10', 'EMA_20', 'DPO_11', 'EMV_1period',
                          'EMV_14', 'Force_index_1', 'Force_index_13', 'Single_EMA', 'Double_EMA', 'EMA_Ratio',
                          'Mass_Index_25', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'positive_flow', 'negative_flow',
                          'Money_Flow_Ratio_14', 'Money_Flow_Index_14', 'close_pctchange', 'volume_pctchange',
                          'NVI_value', 'NVI', 'NVI_signal', 'OBV', 'OBV_signal_65', 'OBV_signal_20', 'PPO',
                          'PPO_signal', 'PPO_hist', 'PVO', 'PVO_signal', 'PVO_hist', 'ROC_10', 'ROC_15', 'ROC_20',
                          'ROC_30', 'RCMA1', 'RCMA2', 'RCMA3', 'RCMA4', 'KST', 'KST_signal_9', 'ROC_40', 'ROC_65',
                          'ROC_75', 'ROC_100', 'RCMA5', 'RCMA6', 'RCMA7', 'RCMA8', 'SpecialK', 'SpecialK_signal_100',
                          'ROC_125', 'ROC_250', 'RSI_5', 'RSI_14', 'LINEARREG_SLOPE_20', 'LINEARREG_SLOPE_100',
                          'STDDEV_10', 'STDDEV_250', 'high_14', 'low_14', '%K_14', '%D_3', 'RIS_high_14', 'RIS_low_14',
                          'StochRSI_14', 'TRIX_15', 'TRIX_signal_9', 'PC', 'abs_PC', 'First_Smoothing_PC_25',
                          'Second_Smoothing_PC_13', 'First_Smoothing_absPC_25', 'Second_Smoothing_absPC_13', 'TSI',
                          'TSI_signal_7', 'max_close_14', 'Percent_Drawdown_14', 'Ulcer_Index_14', 'BP', 'TR', 'UO',
                          'Plus_VM', 'Minus_VM', 'Plus_VM_14', 'Minus_VM_14', 'H-L', 'H-pC', 'L-pC', 'TR_14',
                          'Plus_VI_14', 'Minus_VI_14', 'WILLR_14']

last_data = ['last_price_1', 'last_price_2', 'last_price_3', 'last_price_5', 'last_price_10', 'last_price_15', 'last_price_30', 'last_price_60', 'last_price_180', 'last_price_252',
             'last_volume_1', 'last_volume_2', 'last_volume_3', 'last_volume_5', 'last_volume_10', 'last_volume_15', 'last_volume_30', 'last_volume_60', 'last_volume_180', 'last_volume_252']



   
if __name__ == "__main__":
    pass