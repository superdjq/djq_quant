from djq_train_model import StcokClassifier
#from djq_risk_control import cal_weights, position_control
import numpy as np
import pandas as pd
import time, os, sys
import djq_data_processor
import chinese_calendar, datetime
import dtshare

class Trader(object):
    BASE_DIR = 'trade/'
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
        self.data_path = config.data_path
        assert set(self.weights.keys()) == set(self.model_names.keys())
        self.securities = set(self.model_names.keys())
        self.report = self.BASE_DIR + self.name + '/report.log'
        # 爬取当天开盘盘面信息，主要获取市值用于加权
        print('Start crawling market infomation')
        self.mkt = dtshare.stock_zh_a_spot()
        # self.mkt = pd.read_csv('E:\\WORK\\quant\\tmp\\mkt.csv')
        self.mkt = self.mkt.set_index('code')
        self.df_pred = self.initial_pred()
        self.df_return = self.initial_env()
        assert len(self.df_pred) == len(self.df_return)
        self.thresholds_u = config.thresholds_u
        self.thresholds_d = config.thresholds_d
        self.steps = config.steps
        if not os.path.isfile(self.BASE_DIR + self.name + '/book.csv'):
            self.create_book()
        self.book = pd.read_csv(self.BASE_DIR + self.name + '/book.csv', index_col=0)
        self.pos = dict(self.book.iloc[-1])
        if self.last_workday() != self.book.index[-1]:
            for xtye in ['D', '5']:
                pass
                #djq_data_processor.stock_update(xtye)
                #djq_data_processor.index_update(xtye)
            self.update()




    def initial_pred(self):
        df_pred = pd.DataFrame()
        df_index = None
        for name, model_name in self.model_names.items():
            df = StcokClassifier(model_name).daily_predict()
            df_index = df.index
            df_pred.loc[:, name] = self.cls_to_weighted_pct(df, model_name)
        if df_index is not None:
            df_pred = df_pred.set_index(df_index)
        print(df_pred)
        return df_pred

    def initial_env(self):
        df_env = pd.DataFrame(index=self.df_pred.index)
        for name, path in self.data_path.items():
            df_stk = pd.read_csv(path, index_col=0, usecols=['date', 'close'])
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
        for name, model_name in self.model_names.items():
            res = StcokClassifier(model_name).daily_predict()
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

    def last_workday(self):
        i = 1
        while not chinese_calendar.is_workday(datetime.date.today()-datetime.timedelta(days=i)):
            i += 1
        return ((datetime.datetime.now()-datetime.timedelta(days=i)).strftime("%Y-%m-%d"))




if __name__ == '__main__' and chinese_calendar.is_workday(datetime.date.today()):
    trade = Trader('test')
    while time.localtime().tm_hour < 16:

        trade.daily_monitor()
        trade.print_to_file('------------------------------------------------------------------------------------------------------------------------------------------')
        time.sleep(600)
