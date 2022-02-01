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
import numpy as np
import pandas as pd
import time, os, sys
import json
import djq_data_processor
import dtshare
import tushare as ts
import djq_crawler




class Trader(object):
    BASE_DIR = os.path.abspath('.') + '/trade/'
    COMM_PCT = 0.001

    def __init__(self, name):
        self.name = name
        sys.path.append(self.BASE_DIR + self.name)
        try:
            import config
        except ModuleNotFoundError:
            raise ImportError('Cannot find the config file')
        self.weights = config.weights
        self.model_names = config.model_names
        self.agents = config.agent
        self.etf_names = config.etf_names
        self.total_cash = config.total_cash
        self.steps = config.steps
        assert set(self.weights.keys()) == set(self.model_names.keys())
        self.securities = set(self.model_names.keys())
        self.report = self.BASE_DIR + self.name + '/report.log'
        djq_data_processor.data_update()
        # create a book of estimated result with local data
        self.df_pred = self.initial_pred()
        print(self.df_pred)
       #  self.df_pred = pd.read_csv('E:/WORK/quant/tmp/new_model_threshold2.csv',index_col=0)
        # create a environment which tells daily stock/index change
        self.df_return = self.initial_env()
        print(self.df_return)
        assert len(self.df_pred) == len(self.df_return)
        if not os.path.isfile(self.BASE_DIR + self.name + '/book.csv'):
            self.create_book()
        self.book = pd.read_csv(self.BASE_DIR + self.name + '/book.csv', index_col=0)
        self.pos = dict(self.book.iloc[-1])
        self.update()

    def initial_pred(self):
        df_pred = pd.DataFrame()
        for name, model_name in self.model_names.items():
            df = pd.DataFrame()
            for i in range(5):
                df[i] = StcokClassifier(model_name).daily_predict(real_time=False).weighted_pct
            df_pred.loc[:, name] = df.mean(1)
        return df_pred

    def initial_env(self):
        df_env = pd.DataFrame(index=self.df_pred.index)
        for name, etf_name in self.etf_names.items():
            try:
                df_stk = djq_data_processor.get_data(etf_name, inx=False)
            except:
                raise ValueError('Cannot find the file!')
            df_stk = df_stk[['date', 'close']]
            df_stk = df_stk.set_index('date')
            df_stk = df_stk.rename(columns={'close': name})
            df_stk = df_stk.sort_values('date')
            # df_stk = df_stk.pct_change() + 1
            df_env = df_env.join(df_stk)
        df_env = df_env.fillna(method='ffill')
        df_env = df_env.fillna(method='backfill')

        return df_env

    def create_book(self):

        self.pos = dict()
        self.pos['total'] = self.total_cash
        for stk in self.securities:
            self.pos[stk+'_value'] = self.total_cash * self.weights[stk]
            self.pos[stk+'_shares'] = 0
            self.pos[stk+'_cash'] = self.total_cash * self.weights[stk]
            self.pos[stk+'_pos'] = 0
        self.book = pd.DataFrame(index=self.df_pred.index, columns=self.pos.keys())
        for i in range(len(self.df_pred)):
            self.pos_change(i)
            self.book.iloc[i] = self.pos
        self.book.to_csv(self.BASE_DIR + self.name + '/book.csv')

    def pos_change(self, i, execute_trade=False):
        """
        :param signal: daily estimated change of each project
        :param ret: the real change of each stock of last trading day
        :param execute_trade:
        :return: None, record daily position change and cumulative profit change in book,
        """
        self.print_to_file(str(dict(self.df_pred.iloc[i])))
        self.pos['total'] = 0
        for stk in self.securities:
            self.pos[stk+'_value'] = self.pos[stk+'_cash'] + self.pos[stk+'_shares'] * dict(self.df_return.iloc[i])[stk]
        sell_book = {}
        buy_book = {}
        for stk in self.securities:
            price = dict(self.df_return.iloc[i])[stk]
            pred_his = list(self.df_pred.iloc[max(0, i-4):i+1][stk])
            pos_his = list(self.book.iloc[max(0, i-5):i][stk+'_pos'])
            pred_his = [0] * (5-len(pred_his)) + pred_his
            pos_his = [0] * (5-len(pos_his)) + pos_his
            observation = np.array(pos_his+pred_his)
            action = self.agents[stk].get_action(observation)
            if action == 2 and self.pos[stk+'_pos'] < self.steps[stk]:
                self.pos[stk+'_pos'] += 1
                total_stock = self.pos[stk+'_value'] * self.pos[stk+'_pos'] / self.steps[stk]
                buy_shares = max(0, (total_stock // (price * 100)) * 100 - self.pos[stk+'_shares'])
                self.pos[stk + '_shares'] += buy_shares
                self.pos[stk+'_cash'] -= buy_shares * price * (1 + self.COMM_PCT)
                self.pos[stk+'_value'] -= buy_shares * price * self.COMM_PCT
                if buy_shares:
                    buy_book[self.etf_names[stk]] = buy_shares

            elif action == 0 and self.pos[stk+'_pos'] > 0:
                self.pos[stk + '_pos'] -= 1
                total_stock = self.pos[stk + '_value'] * self.pos[stk + '_pos'] / self.steps[stk]
                sell_shares = max(0, self.pos[stk + '_shares'] - (total_stock // (price * 100)) * 100)
                self.pos[stk + '_shares'] -= sell_shares
                self.pos[stk + '_cash'] += sell_shares * price * (1 - self.COMM_PCT)
                self.pos[stk + '_value'] -= sell_shares * price * self.COMM_PCT
                if sell_shares:
                    sell_book[self.etf_names[stk]] = sell_shares
            self.pos['total'] += self.pos[stk+'_value']

            if execute_trade:
                while time.strftime('%H%M') < '0926':
                    time.sleep(60)
                for name, share in sell_book.items():
                    crawler = djq_crawler.Crawler()
                    crawler.sell(name, share)
                    crawler.close()
                for name, share in buy_book.items():
                    crawler = djq_crawler.Crawler()
                    crawler.buy(name, share)
                    crawler.close()



    def update(self):
        """
        when first run the manage process in a day, update the current position and
        profit after download data of yesterday
        Change your real trade position as the result shows
        :return: None
        """
        if self.book.index.values[-1] == self.df_pred.index.values[-1]:
            self.show_pos()
            self.print_to_file('--------------------------------------------------------------------'
                               '----------------------------------------------------------------------')
            return
        for i in range(len(self.book), len(self.df_pred)):
            date = self.df_pred.index[i]
            self.print_to_file('Position Update for trade: {} on date: {}'.format(self.name, date))
            self.pos_change(i, execute_trade=True)
            # self.cal_profit(dict(self.df_return.iloc[i]))
            # self.pos['cum_profit'] -= sum(abs(pd.Series(self.pos)[self.securities] - self.book.iloc[i - 1,
            #                                                      :len(self.securities)])) * self.COMM_PCT
            self.book.loc[date] = self.pos
            self.show_pos()
            self.print_to_file('Cumulative profit is {:.2f}%'.format(100*(self.pos['total'] / self.total_cash-1)))
            self.print_to_file('-----------------------------------------------------------------------'
                               '-------------------------------------------------------------------')
        self.book.to_csv(self.BASE_DIR + self.name + '/book.csv')

    def show_pos(self):
        if len(self.book):
            for stk in self.securities:
                self.print_to_file('stk:{} with shares:{}'.format(stk, self.book.iloc[-1][stk+'_shares']))

    def daily_monitor(self):
        """
        Monitor real time data when market is open
        When result breaks the threshold, show position change warning as a possibility of tomorrow position change,
        so you can prepare in advance
        :return: None
        """
        for name, model_name in self.model_names.items():
            res = StcokClassifier(model_name).daily_predict(real_time=True)
            score = res.weighted_pct.values[-1]

            self.print_to_file(time.strftime('%Y-%m-%d %H:%M:%S') +
                               " score for project:{} is: {}".format(model_name, score))

            if time.localtime().tm_hour > 12:
                pred_his = list(res.weighted_pct)[-5:]
                pos_his = list(self.book[name + '_pos'])[-5:]
                pred_his = [0] * (5 - len(pred_his)) + pred_his
                pos_his = [0] * (5 - len(pos_his)) + pos_his
                observation = np.array(pos_his + pred_his)
                action = self.agents[name].get_action(observation)
                if action == 2 and self.pos[name+'_pos'] < self.steps[name]:
                    etf_price = float(ts.get_realtime_quotes(self.etf_names[name]).price.values[0])
                    total_stock = self.pos[name + '_value'] * (self.pos[name + '_pos'] + 1) / self.steps[name]
                    buy_shares = max(0, (total_stock // (etf_price * 100)) * 100 - self.pos[name + '_shares'])
                    if buy_shares:
                        self.print_to_file(' You should buy this ETF {} shares'.format(buy_shares))
                elif action == 0 and self.pos[name+'_pos'] > 0:
                    etf_price = float(ts.get_realtime_quotes(self.etf_names[name]).price.values[0])
                    total_stock = self.pos[name + '_value'] * (self.pos[name + '_pos'] - 1) / self.steps[name]
                    sell_shares = max(0, self.pos[name + '_shares'] - (total_stock // (etf_price * 100)) * 100)
                    self.print_to_file(' You should sell this ETF {} shares'.format(sell_shares))

    def print_to_file(self, info):
        with open(self.report, 'a+') as f:
            f.write(info + '\n')
        print(info)









if __name__ == '__main__':# and chinese_calendar.is_workday(datetime.date.today()):
    trade = Trader('main_etf_trader_2021')
    #trade = Trader('test')