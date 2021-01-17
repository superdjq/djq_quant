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
from sqlalchemy import create_engine
import datetime, chinese_calendar
import requests
import akshare as ak


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
        train_x, train_y, test_x, test_y, df_test = prepare_single_df_to_lstm(df, xlst, window=window, target_day=target_day, noclose=noclose)
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


def stock_update(df_all_stock_daily=None):
    '''
    Update your local stock data csv
    :param typ:
        typ: data frequency 'D'-daily, 'M'-monthly, 'Y'-yearly,
                            '5'-every 5 minutes, same as '15','30','60'
    :return: Nothing
    '''
    if df_all_stock_daily is None:
        df_all_stock_daily = dt.stock_zh_a_spot()
    if zsys.use_mysql:
        engine = create_engine("mysql+mysqlconnector://%s:%s@%s:%s/%s?charset=utf8"%(zsys.mysql_user,
                                                                              zsys.mysql_password,
                                                                              zsys.mysql_host,
                                                                              zsys.mysql_port, 'stk'))
    n = len(df_all_stock_daily['code'])
    # for i,xc in enumerate(stkPool['code']):
    file_path = zsys.rdatCN
    for i, rx in df_all_stock_daily.iterrows():
        code = rx['code']
        print("\n", i, "/", n, '@', code, rx['name'], ",@", file_path if not zsys.use_mysql else
              'schema: stk, table: %s' % code)
        tim0, fss = '2010-01-01', file_path + code + '.csv'
        if zsys.use_mysql:
            xfg = engine.has_table(code)
        else:
            xfg = os.path.exists(fss) and (os.path.getsize(fss) > 0)
        if xfg:
            xd0 = pd.read_sql_table(code, engine) if zsys.use_mysql else pd.read_csv(fss, index_col=False, encoding='utf8')
            tim0 = list(xd0.date)[-1]
        print('\t', xfg, ",", tim0)
    # -----------
        try:
            xdk = ts.get_k_data(code, index=False, start=tim0, end=None)
        # -------------
            if len(xdk) > 0:
                xdk = xdk[zsys.ohlcDVLst]
                if zsys.use_mysql:
                    if xfg:
                        if len(xdk) > 1:
                            xdk.iloc[1:].to_sql(code, engine, if_exists='append', index=False)
                    else:
                        xdk.to_sql(code, engine, index=False)
                else:
                    if xfg:
                        if len(xdk) > 1:
                            xdk.iloc[1:].to_csv(fss, index=False, encoding='utf8', mode='a', header=0)
                    else:
                        xdk.to_csv(fss, index=False, encoding='utf8')

        except IOError:
            print('error')
            # pass  # skip,error
    # remove ex-divdend stock data, which will be downloaded again the next day
    for code, name in zip(df_all_stock_daily.code, df_all_stock_daily.name):
        if name.startswith('XR') or name.startswith('XD') or name.startswith('DR'):
            if zsys.use_mysql:
                if engine.has_table(code):
                    engine.execute('DROP TABLE `' + code + '`')
            else:
                fss = file_path + code + '.csv'
                if os.path.exists(fss):
                    os.remove(fss)





def index_update():
    """
    Update your local index data csv
    :param typ:
        typ: data frequency 'D'-daily, 'M'-monthly, 'Y'-yearly,
                            '5'-every 5 minutes, same as '15','30','60'
    :return: Nothing
    """
    df_all_index_daily = dt.index_stock_info()
    if zsys.use_mysql:
        engine = create_engine("mysql+mysqlconnector://%s:%s@%s:%s/%s?charset=utf8" % (zsys.mysql_user,
                                                                                zsys.mysql_password,
                                                                                zsys.mysql_host,
                                                                                zsys.mysql_port, 'inx'))
    n = len(df_all_index_daily['index_code'])
    # for i,xc in enumerate(stkPool['code']):
    file_path = zsys.rdatCNX
    for i, rx in df_all_index_daily.iterrows():
        code = rx['index_code']
        print("\n", i, "/", n, '@', code, rx['display_name'], ",@", file_path if not zsys.use_mysql else
              'schema: index, table: %s' % code)
        tim0, fss = '2010-01-01', file_path + code + '.csv'
        if zsys.use_mysql:
            xfg = engine.has_table(code)
        else:
            xfg = os.path.exists(fss) and (os.path.getsize(fss) > 0)
        if xfg:
            xd0 = pd.read_sql_table(code, engine) if zsys.use_mysql else pd.read_csv(fss, index_col=False, encoding='utf8')
            tim0 = list(xd0.date)[-1]

        print('\t', xfg, ",", tim0)
        # -----------
        try:
            xdk = ts.get_k_data(code, index=True, start=tim0, end=None)
            if len(xdk) > 0:
                xdk = xdk[zsys.ohlcDVLst]
                if zsys.use_mysql:
                    if xfg:
                        if len(xdk) > 1:
                            xdk.iloc[1:].to_sql(code, engine, if_exists='append', index=False)
                    else:
                        xdk.to_sql(code, engine, index=False)
                else:
                    if xfg:
                        if len(xdk) > 1:
                            xdk.iloc[1:].to_csv(fss, index=False, encoding='utf8', mode='a', header=0)
                    else:
                        xdk.to_csv(fss, index=False, encoding='utf8')

        except IOError:
            pass  # skip,error

def etf_update():
    """
    Update your local index data csv
    :param typ:
        typ: data frequency 'D'-daily, 'M'-monthly, 'Y'-yearly,
                            '5'-every 5 minutes, same as '15','30','60'
    :return: Nothing
    """
    df_all_etf_daily = ak.fund_etf_category_sina('ETF基金')
    if zsys.use_mysql:
        engine = create_engine("mysql+mysqlconnector://%s:%s@%s:%s/%s?charset=utf8" % (zsys.mysql_user,
                                                                                zsys.mysql_password,
                                                                                zsys.mysql_host,
                                                                                zsys.mysql_port, 'stk'))
    n = len(df_all_etf_daily['symbol'])
    # for i,xc in enumerate(stkPool['code']):
    file_path = zsys.rdatCNX
    for i, rx in df_all_etf_daily.iterrows():
        code = rx['symbol'][2:]
        print("\n", i, "/", n, '@', code, rx['name'], ",@", file_path if not zsys.use_mysql else
              'schema: stk, table: %s' % code)
        tim0, fss = '2010-01-01', file_path + code + '.csv'
        if zsys.use_mysql:
            xfg = engine.has_table(code)
        else:
            xfg = os.path.exists(fss) and (os.path.getsize(fss) > 0)
        if xfg:
            xd0 = pd.read_sql_table(code, engine) if zsys.use_mysql else pd.read_csv(fss, index_col=False, encoding='utf8')
            tim0 = list(xd0.date)[-1]

        print('\t', xfg, ",", tim0)
        # -----------
        try:
            xdk = ts.get_k_data(code, index=False, start=tim0, end=None)
            if len(xdk) > 0:
                xdk = xdk[zsys.ohlcDVLst]
                if zsys.use_mysql:
                    if xfg:
                        if len(xdk) > 1:
                            xdk.iloc[1:].to_sql(code, engine, if_exists='append', index=False)
                    else:
                        xdk.to_sql(code, engine, index=False)
                else:
                    if xfg:
                        if len(xdk) > 1:
                            xdk.iloc[1:].to_csv(fss, index=False, encoding='utf8', mode='a', header=0)
                    else:
                        xdk.to_csv(fss, index=False, encoding='utf8')

        except IOError:
            pass  # skip,error









def last_workday():
    i = 1
    while not chinese_calendar.is_workday(datetime.date.today()-datetime.timedelta(days=i)) and \
        datetime.date.weekday(datetime.date.today()-datetime.timedelta(days=i)) > 4:
        i += 1
    return ((datetime.datetime.now()-datetime.timedelta(days=i)).strftime("%Y-%m-%d"))


def data_update():
    """
    Keep your local or mysql database up to date
    Create Schemas:{'stk', 'inx'} in your database
    :return:
    """
    need_update = True
    if zsys.use_mysql:
        db = 'inx'
        engine = create_engine("mysql+mysqlconnector://%s:%s@%s:%s/%s?charset=utf8" % (zsys.mysql_user,
                                                                                zsys.mysql_password,
                                                                                zsys.mysql_host,
                                                                                zsys.mysql_port, db))
        if engine.has_table('399300') and list(pd.read_sql_table('399300', engine).date)[-1] >= last_workday():
            need_update = False
    else:
        path = zsys.rdatCNX + '399300.csv'
        if os.path.exists(path) and list(pd.read_csv(path).date)[-1] == last_workday():
            need_update = False
    if need_update:
        stock_update()
        index_update()
        etf_update()

def get_data(code, inx=False):
    """
    :param code: str, China stock or index code number of length 6
    :param inx: bool, whether the code is the number of index
    :return: pandas.DataFrame, date&ohlcv
    """
    try:
        if zsys.use_mysql:
            db = 'inx' if inx else 'stk'
            engine = create_engine("mysql+mysqlconnector://%s:%s@%s:%s/%s?charset=utf8" % (zsys.mysql_user,
                                                                                            zsys.mysql_password,
                                                                                            zsys.mysql_host,
                                                                                            zsys.mysql_port, db))
            df = pd.read_sql_table(code, engine)
        else:
            df = pd.read_csv((zsys.rdatCNX if inx else zsys.rdatCN) + code + '.csv')
    except:
        print('Fail to load local data, try to crawl online stock data')
        try:
            df = ts.get_k_data(code, start='2010-01-01', index=inx)[zsys.ohlcDVLst]
        except:
            raise ValueError('Cannot get data')
    return df

def get_all_etf_code():
    df_all_index_daily = dt.index_stock_info()
    URL = "http://www.fundsmart.com.cn/api/fund.list.data.php?d=&t=3&i={}"
    etf_code_list = []
    for code in df_all_index_daily.index_code.values:
        ret = requests.get(URL.format(code)).json()['list']
        for info in ret:
            etf_code_list.append(info['ticker'])
    return etf_code_list





if __name__ == '__main__':
    #stock_update()
    #index_update()
    etf_update()



