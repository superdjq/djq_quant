"""
The module "djq_data_processor" includes China stock data crawling,
prepare data set for lstm learning, and data washing process
"""

import pandas as pd
import numpy as np
import tushare as ts
import dtshare as dt
import os
import djq_talib
import zsys
from sklearn import preprocessing


def prepare_single_df_to_lstm(df, xlst, window=30, classify=(2, 0), target_day=5, split_date='2019-01-01', noclose=True):
    df = djq_talib.get_all_finanical_indicators(df, divided_by_close=noclose)
    get_label(df, target_day, pct_change=True)
    xlst = (zsys.ohlVLst if noclose else zsys.ohlcVLst) + xlst
    print(df.head())
    df.dropna(inplace=True)

    if classify is not None:
        transfer_label_to_classification(df, classify)

    df.to_csv('tmp/data.csv')
    total_x = []
    total_y = []
    scaler = preprocessing.StandardScaler()
    #df[xlst] = scaler.fit_transform(df[xlst])
    for i in range(window-1, len(df)):
        tmp = df.iloc[i-window+1: i+1].copy()
        xdat = np.array(tmp[xlst])
        xdat = scaler.fit_transform(xdat)
        total_x.append(xdat)
        total_y.append(df.iloc[i]['y'])
    train_split = max(len(df[df.date<split_date])-window+1, 0)
    total_x = np.array(total_x)
    train_x = total_x[:train_split]
    test_x = total_x[train_split:]
    if classify is not None:
        total_y = pd.get_dummies(total_y).values
    else:
        total_y = np.array(total_y)
    #total_y = np.array(total_y)
    train_y = total_y[:train_split]
    test_y = total_y[train_split:]
    df_test = df[df.date >= split_date].tail(len(test_y))
    return train_x, train_y, test_x, test_y, df_test

def prepare_predict_to_lstm(inx='sz50', xlst=zsys.ohlcVLst, window=30, noclose=True, target_day=5):
    finx = 'TQDat/data/stk_' + inx + '.csv'
    inx_df = pd.read_csv(finx, dtype={'code': str}, encoding='GBK')
    clst = list(inx_df['code'])
    test_total_x = []
    df_total_test = pd.DataFrame()
    rss = zsys.rdatCN
    for code in clst:
        print('Now processing stk: {}'.format(code))
        df = pd.read_csv(rss+code+'.csv')
        df['code'] = code
        train_x, train_y, test_x, test_y, df_test = prepare_single_df_to_lstm(df, xlst, window=window, noclose=noclose, target_day=target_day)
        test_total_x += test_x.tolist()
        df_total_test = df_total_test.append(df_test)
    return np.array(test_total_x), df_total_test

def prepare_inx_to_lstm(inx='sz50', xlst=zsys.ohlcVLst, window=30, one_hot=True, classify_params=('n', 96, 105), target_day=5, active_date='2002-01-01', need_shuffle=False, noclose=True):
    finx = 'TQDat/data/stk_' + inx + '.csv'
    inx_df = pd.read_csv(finx, dtype={'code': str}, encoding='GBK')
    clst = list(inx_df['code'])
    train_total_x = []
    train_total_y = []
    test_total_x = []
    test_total_y = []
    df_total_test = pd.DataFrame()
    rss = zsys.rdatCN
    for code in clst:
        print('Now processing stk: {}'.format(code))
        df = pd.read_csv(rss+code+'.csv')
        df['code'] = code
        if active_date is not None: df = df[df.date >=active_date]
        train_x, train_y, test_x, test_y, df_test = prepare_single_df_to_lstm(df, xlst, window=window, one_hot=one_hot, classify_params=classify_params, target_day=target_day, noclose=noclose)
        train_total_x += train_x.tolist()
        train_total_y += train_y.tolist() if type(train_y) == np.ndarray else train_y
        test_total_x += test_x.tolist()
        test_total_y += test_y.tolist() if type(test_y) == np.ndarray else test_y
        df_total_test = df_total_test.append(df_test)
    shuffle_train = np.arange(len(train_total_x))
    shuffle_test = np.arange(len(test_total_x))
    if need_shuffle:
        shuffle_train = np.random.permutation(np.arange(len(train_total_x)))
        shuffle_test = np.random.permutation(np.arange(len(test_total_x)))
    df_total_test = df_total_test.iloc[shuffle_test]
    return np.array(train_total_x)[shuffle_train], np.array(train_total_y)[shuffle_train], np.array(test_total_x)[shuffle_test], np.array(test_total_y)[shuffle_test], df_total_test


def transfer_label_to_classification(df, classify=(2,0)):
    if classify[0] == 2:
        assert len(classify) == 2
        df['y'] = np.where(df['y'] > classify[1], 1, 0)
    else:
        assert len(classify) == 3
        tmp = df['y'].copy()
        df['y'] = 0
        bottom = classify[1]
        step = (classify[2] - classify[1]) / (classify[0]-2)
        for i in range(1, classify[0]):
            df.loc[tmp >= bottom, 'y'] = i
            bottom += step
        df['y_pct_change'] = tmp

def get_label(df, target_day=5, pct_change=True):
    df['y'] = df.close.shift(-target_day)
    if pct_change:
        df['y'] = 100 * (df['y'] - df.close) / df.close

def stock_update(typ='D'):
    '''
    Update your local stock data csv
    :param typ:
        typ: data frequency 'D'-daily, 'M'-monthly, 'Y'-yearly,
                            '5'-every 5 minutes, same as '15','30','60'
    :return: Nothing
    '''

    df_all_stock_daily = dt.stock_zh_a_spot()
    n = len(df_all_stock_daily['code'])
    # for i,xc in enumerate(stkPool['code']):
    file_path = zsys.rdatCN if typ == 'D' else zsys.rdatMin0 + 'm' + '0'*(2-len(typ)) + typ + '/stk/'
    for i, rx in df_all_stock_daily.iterrows():
        code = rx['code']
        print("\n", i, "/", n, '@', code, rx['name'], ",@", file_path)
        tim0, fss = '1994-01-01', file_path + code + '.csv'
        xd0, xd = [], []
        xfg = os.path.exists(fss) and (os.path.getsize(fss) > 0)
        if xfg:
            xd0, tim0 = df_rdcsv_tim0(fss, 'date', tim0)

        print('\t', xfg, typ, fss, ",", tim0)
    # -----------
        try:
            xdk = ts.get_k_data(code, index=False, start=tim0, end=None, ktype=typ);
            xd = xdk
        # -------------
            if len(xd) > 0:
                xd = xdk[zsys.ohlcDVLst]
                xd = df_xappend(xd, xd0, 'date')
                xd = xd.sort_values(['date'], ascending=False)
                xd.to_csv(fss, index=False, encoding='utf8')

        except IOError:
            pass  # skip,error


def index_update(typ='D'):
    '''
    Update your local index data csv
    :param typ:
        typ: data frequency 'D'-daily, 'M'-monthly, 'Y'-yearly,
                            '5'-every 5 minutes, same as '15','30','60'
    :return: Nothing
    '''
    df_all_index_daily = dt.index_stock_info()
    n = len(df_all_index_daily['index_code'])
    # for i,xc in enumerate(stkPool['code']):
    file_path = zsys.rdatCNX if typ == 'D' else zsys.rdatMin0 + 'm' + '0'*(2-len(typ)) + typ + '/inx/'
    for i, rx in df_all_index_daily.iterrows():
        code = rx['index_code']
        print("\n", i, "/", n, '@', code, rx['display_name'], ",@", file_path)
        tim0, fss = '1994-01-01', file_path + code + '.csv'
        xd0, xd = [], []
        xfg = os.path.exists(fss) and (os.path.getsize(fss) > 0)
        if xfg:
            xd0, tim0 = df_rdcsv_tim0(fss, 'date', tim0)

        print('\t', xfg, typ, fss, ",", tim0)
        # -----------
        try:
            xdk = ts.get_k_data(code, index=True, start=tim0, end=None, ktype=typ);
            xd = xdk
            # -------------
            if len(xd) > 0:
                xd = xdk[zsys.ohlcDVLst]
                xd = df_xappend(xd, xd0, 'date')
                xd = xd.sort_values(['date'], ascending=False)
                xd.to_csv(fss, index=False, encoding='utf8')

        except IOError:
            pass  # skip,error






def df_rdcsv_tim0(fss, ksgn, tim0):
    # xd0= pd.read_csv(fss,index_col=False,encoding='gbk')
    xd0 = pd.read_csv(fss, index_col=False, encoding='utf8')
    # print('\nxd0\n',xd0.head())
    if (len(xd0) > 0):
        # xd0=xd0.sort_index(ascending=False);
        # xd0=xd0.sort_values(['date'],ascending=False);
        xd0 = xd0.sort_values([ksgn], ascending=True)
        # print('\nxd0\n',xd0)
        xc = xd0.index[-1]
        _xt = xd0[ksgn][xc]
        s2 = str(_xt)
        # print('\nxc,',xc,_xt,'s2,',s2)
        if s2 != 'nan':
            tim0 = s2.split(" ")[0]

            #
    return xd0, tim0


def df_xappend(df, df0, ksgn, num_round=3, vlst=zsys.ohlcDVLst):
    if (len(df0) > 0):
        df2 = df0.append(df)
        df2 = df2.sort_values([ksgn], ascending=True)
        df2.drop_duplicates(subset=ksgn, keep='last', inplace=True)
        # xd2.index=pd.to_datetime(xd2.index);xd=xd2
        df = df2

    #
    df = df.sort_values([ksgn], ascending=False)
    df = np.round(df, num_round)
    df2 = df[vlst]
    #
    return df2


if __name__ == '__main__':
    for xtye in ['D','5']:
        # data_update(xtye, rdat)
        stock_update(xtye)
        index_update(xtye)


