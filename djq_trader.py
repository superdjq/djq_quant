"""
The module "djq_trader" helpers to build a trade manager to manage the portfolio of
several projects trained by module "djq_train_model"
You should set a config file in folder "/trade/manager_name", include project names,
weights of the portfolio, up/down threshold, which can be calculated by utils
After manager initialized, a book will be created in "/trade/manager_name", which records
daily positions following the strategy, you can set pos to 0, and total profit to 1 when
you start to make real deal
"""
from djq_train_model import StcokClassifier
#from djq_risk_control import cal_weights, position_control
import numpy as np
import pandas as pd
import time, os, sys
import djq_data_processor
import dtshare


class Trader(object):
    BASE_DIR = os.path.abspath('.') + '/trade/'
    COMM_PCT = 0.001

    def __init__(self, name):
        self.name = name
        sys.path.append(self.BASE_DIR + self.name)
        try:
            import config
        except:
            raise ImportError('Cannot find the config file')
        self.weights = config.weights
        self.model_names = config.model_names
        self.thresholds_u = config.thresholds_u
        self.thresholds_d = config.thresholds_d
        self.steps = config.steps
        assert set(self.weights.keys()) == set(self.model_names.keys())
        self.securities = set(self.model_names.keys())
        self.report = self.BASE_DIR + self.name + '/report.log'
        # 爬取当天开盘盘面信息，主要获取市值用于加权
        print('Start crawling market infomation')
        self.mkt = dtshare.stock_zh_a_spot()
        time.sleep(20)
        # self.mkt = pd.read_csv('E:\\WORK\\quant\\tmp\\mkt.csv')
        djq_data_processor.data_update()
        self.mkt = self.mkt.set_index('code')
        # create a book of estimated result with local data
        self.df_pred = self.initial_pred()
        # create a environment which tells daily stock/index change
        self.df_return = self.initial_env()
        assert len(self.df_pred) == len(self.df_return)
        if not os.path.isfile(self.BASE_DIR + self.name + '/book.csv'):
            self.create_book()
        self.book = pd.read_csv(self.BASE_DIR + self.name + '/book.csv', index_col=0)
        self.pos = dict(self.book.iloc[-1])

        self.update()

    def initial_pred(self):
        df_pred = pd.DataFrame()
        df_index = None
        for name, model_name in self.model_names.items():
            df = StcokClassifier(model_name).daily_predict(real_time=False)
            df_index = df.index
            df_pred.loc[:, name] = self.cls_to_weighted_pct(df, model_name)
        if df_index is not None:
            df_pred = df_pred.set_index(df_index)
        return df_pred

    def initial_env(self):
        df_env = pd.DataFrame(index=self.df_pred.index)
        for name, model_name in self.model_names.items():
            try:
                df_stk = djq_data_processor.get_data(StcokClassifier(model_name).inx, inx=True)
            except:
                raise ValueError('Cannot find the file!')
            df_stk = df_stk[['date', 'close']]
            df_stk = df_stk.set_index('date')
            df_stk = df_stk.rename(columns={'close': name})
            df_stk = df_stk.sort_values('date')
            df_stk = df_stk.pct_change() + 1
            df_env = df_env.join(df_stk)
        df_env = df_env.fillna(method='ffill')
        df_env = df_env.fillna(method='backfill')

        return df_env

    def create_book(self):
        self.book = pd.DataFrame(index=self.df_pred.index, columns=list(self.securities) + ['cum_profit'])
        self.pos = dict(zip(self.securities, [0] * len(self.securities)))
        self.pos['cum_profit'] = 1
        for i in range(len(self.df_pred)):
            self.pos_change(dict(self.df_pred.iloc[i]), dict(self.df_return.iloc[i]))
            # self.cal_profit(dict(self.df_return.iloc[i]))
            # self.pos['cum_profit'] -= sum(abs(pd.Series(self.pos)[self.securities] - self.book.iloc[i - 1, :len(self.securities)])) * self.COMM_PCT
            self.book.iloc[i] = self.pos
        # self.book.iloc[0]['cum_profit'] = 1 - sum(self.book.iloc[0,:len(self.securities)]) * self.COMM_PCT
        #for i in range(1,len(self.book)):
            #self.book.iloc[i]['cum_profit'] = self.cal_profit(dict(self.df_return.iloc[i])) \
        #                                        - sum(abs(self.book.iloc[i,:len(self.securities)]-self.book.iloc[i-1,:len(self.securities)])) * self.COMM_PCT
        self.book.to_csv(self.BASE_DIR + self.name + '/book.csv')

    def pos_change(self, signal, ret):
        """
        :param signal: daily estimated change of each project
        :param ret: the real change of each stock of last trading day
        :return: None, record daily position change and cumulative profit change in book,
        """
        self.print_to_file(str(signal))
        res = 0
        for stk in self.securities:
            res += self.pos[stk] * ret[stk] + (self.weights[stk] - self.pos[stk])
        self.pos['cum_profit'] *= res
        tmp = self.pos.copy()
        for stk in self.securities:
            if signal[stk] >= self.thresholds_u[stk]:
                self.pos[stk] = min(self.pos[stk] + self.weights[stk] / self.steps[stk], self.weights[stk])
            elif signal[stk] <= self.thresholds_d[stk]:
                self.pos[stk] = max(self.pos[stk] - self.weights[stk] / self.steps[stk], 0)
        self.pos['cum_profit'] -= sum([abs(self.pos[stk] - tmp[stk]) for stk in self.securities]) * self.COMM_PCT * \
                                  self.pos['cum_profit']

    def update(self):
        """
        when first run the manage process in a day, update the current position and
        profit after download data of yesterday
        Change your real trade position as the result shows
        :return: None
        """
        if len(self.book) == len(self.df_pred):
            self.show_pos()
            return
        for i in range(len(self.book), len(self.df_pred)):
            date = self.df_pred.index[i]
            self.print_to_file('Position Update for trade: {} on date: {}'.format(self.name, date))
            self.pos_change(dict(self.df_pred.iloc[i]), dict(self.df_return.iloc[i]))
            # self.cal_profit(dict(self.df_return.iloc[i]))
            # self.pos['cum_profit'] -= sum(abs(pd.Series(self.pos)[self.securities] - self.book.iloc[i - 1,
            #                                                      :len(self.securities)])) * self.COMM_PCT
            self.book.loc[date] = self.pos
            self.show_pos()
            self.print_to_file('Cumulative profit is {}'.format(self.pos['cum_profit']))
            self.print_to_file('------------------------------------------------------------------------------------------------------------------------------------------')
        self.book.to_csv(self.BASE_DIR + self.name + '/book.csv')

    def cls_to_weighted_pct(self, df, pjNam):
        """
        The estimated result is a class interval number, change the class to corresponding pct change
        Calculate the index pct change using the market value weights
        :param df: pandas.DataFrame, estimated result
        :param pjNam: str, project name
        :return: float, estimated change of index
        """
        book = StcokClassifier(pjNam).book
        df2 = df.astype(float)
        for stk in df.columns:
            df2.loc[:][stk] = [book[book.code == stk]['tier' + str(int(c))].values[0] for c in df.loc[:][stk]]
        return np.average(df2, weights=self.mkt.loc[df.columns]['mktcap'], axis=1)

    def show_pos(self):
        if len(self.book):
            for stk in self.securities:
                self.print_to_file('stk:{} with position:{}'.format(stk, self.book.iloc[-1][stk]))

    def daily_monitor(self):
        """
        Monitor real time data when market is open
        When result breaks the threshold, show position change warning as a possibility of tomorrow position change,
        so you can prepare in advance
        :return: None
        """
        for name, model_name in self.model_names.items():
            res = StcokClassifier(model_name).daily_predict(real_time=True)
            score = self.cls_to_weighted_pct(res, model_name)[-1]

            self.print_to_file(time.strftime('%Y-%m-%d %H:%M:%S') + " score for project:{} is: {}".format(model_name, score))
            if time.localtime().tm_hour > 12:
                if score >= self.thresholds_u[name] and self.pos[name] < self.weights[name]:
                    self.print_to_file(' You should buy this ETF to pos: {}%'.format(100*(self.pos[name]+self.weights[name]/self.steps[name])))
                elif score <= self.thresholds_d[name] and self.pos[name] > 0:
                    self.print_to_file(' You should sell this ETF to pos: {}%'.format(100*(max(0,self.pos[name]-self.weights[name]/self.steps[name]))))

    def print_to_file(self, info):
        with open(self.report, 'a+') as f:
            f.write(info + '\n')
        print(info)


if __name__ == '__main__':# and chinese_calendar.is_workday(datetime.date.today()):
    trade = Trader('main_etf_trader')
    while time.localtime().tm_hour < 16:

        trade.daily_monitor()
        trade.print_to_file('------------------------------------------------------------------------------------------------------------------------------------------')
        time.sleep(600)
