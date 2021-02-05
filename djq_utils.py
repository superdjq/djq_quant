import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from djq_train_model import StcokClassifier
import djq_data_processor
import matplotlib.ticker as ticker
import zsys
import time
from djq_trader import Trader

def mkt_cmp(df, mkt='399300',start_date='2020-01-01'):
    '''
    Draw the picture of portfolio price chart, compared with market
    Draw the linear regression line of portfolio with market, show the beta/alpha value
    :param df: pandas.Series, index = date, time series data of NPV of the portfolio
    :param mkt: str with length=6, China stock index code number
    :param start_date: 'YYYY-mm-dd'
    :return: None
    '''
    assert type(df) == pd.Series
    plt.figure(figsize=(10, 8))
    df.index = pd.to_datetime(df.index)
    df_mkt = pd.read_csv(zsys.rdatCNX + mkt + ".csv", parse_dates=['date'])
    # df_mkt.date = df_mkt.date.dt.strftime('%Y/%m/%d')
    df_mkt = df_mkt.set_index('date')
    if set(df.index) & set(df_mkt.index) == set([]):
        print('Input data error')
        return
    df = df[df.index>=start_date]
    df_mkt = df_mkt.loc[list(df.index)]
    df_mkt['close'] = df_mkt['close'] / df_mkt['close'][0]
    ax1 = plt.subplot(2,1,1)
    #ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
    plt.plot(df)
    plt.plot(df_mkt.close)
    plt.legend(['portfolio_profit','market_profit'], loc='best')
    #plt.xticks(rotation=330)

    df_chg = df.pct_change().dropna()
    mkt_chg = df_mkt.close.pct_change().dropna()
    plt.subplot(2, 1, 2)
    plt.scatter(mkt_chg, df_chg)

    cov_a_b = np.cov(df_chg, mkt_chg)[0][1]
    beta = cov_a_b / mkt_chg.var()
    alpha = df_chg.mean() - beta * mkt_chg.mean()
    x = np.linspace(min(mkt_chg), max(mkt_chg), 50)
    y1 = beta * x + alpha
    plt.plot(x, y1, color='red')
    plt.title('Security Characteristic Line with beta=%.4f, alpha=%.4f'%(beta, alpha))
    plt.show()


class stock_env(object):
    '''
    Create an environment to generate daily stock change and predict value,
    - reset(): reset the environment to the first day
    - step(): return daily observation, real price change of last day, and done as flag
    '''
    def __init__(self, model_name, etf_name, window=5, mode='train', position_split=1):
        self.model = StcokClassifier(model_name)
        self.etf = etf_name
        self.df_pred = self.model.daily_predict(real_time=False).weighted_pct
        self.df_env = self.initial_env()
        self.n_action = 3
        self.window = window
        self.pos = [0] * window
        self.end = len(self.df_env)
        self.commission = 0.001
        self.done = False
        self.position_split = position_split
        self.mode = mode


    def initial_env(self):
        df_env = pd.DataFrame(index=self.df_pred.index)
        df_stk = djq_data_processor.get_data(self.etf)
        df_stk = df_stk.set_index('date')
        df_env = df_env.join(df_stk.close)
        df_env = df_env.fillna(method='ffill')
        df_env = df_env.fillna(method='backfill')
        return df_env

    def reset(self):
        self.cash = 100000
        self.total_value = 100000
        self.pos = [0] * self.window
        self.observation = [0] * self.window
        self.i = 0 if self.mode != 'test' else len(self.df_env) // 2
        self.observation[-1] = self.df_pred.iloc[self.i]
        self.shares = 0
        self.end = len(self.df_env) if self.mode != 'train' else len(self.df_env) // 2
        self.done = False
        # self.action_memory = np.ones((len(self.df_pred), self.n_action)) / self.n_action
        # return np.concatenate([self.df_pred.iloc[self.pos].values, self.action_memory[self.pos]])
        return np.array(self.pos + self.observation)

    def step(self, action):
        price = self.df_env.close.iloc[self.i]
        tmp = self.total_value
        self.total_value = self.cash + price * self.shares
        pos = self.pos[-1]
        if action == 2 and pos < self.position_split:
            pos += 1
            total_stock = self.total_value * pos / self.position_split
            buy_shares = max(0, (total_stock // (price * 100)) * 100 - self.shares)
            self.shares += buy_shares
            self.cash -= buy_shares * price * (1 + 0.001)
            self.total_value -= buy_shares * price * 0.001

        elif action == 0 and pos > 0:
            pos -= 1
            total_stock = self.total_value * pos / self.position_split
            sell_shares = max(0, self.shares - (total_stock // (price * 100)) * 100)
            self.shares -= sell_shares
            self.cash += sell_shares * price * (1 - 0.001)
            self.total_value -= sell_shares * price * 0.001
        self.pos = self.pos[1:] + [pos]
        self.i += 1
        done = self.i >= self.end
        if not done:
            self.observation = self.observation[1:] + [self.df_pred.iloc[self.i]]
        return np.array(self.pos + self.observation), self.total_value - tmp, done, {}

def Monte_Carlo_Simulation(env, threshold_u, threshold_d, pos_step=1, mode='train', need_plot=False, ret_his=False):
    '''
    Simulate trade strategy with simply buy if the price above the up_threshold, and
    sell if price below the down_threshold, and show the total return
    :param env: sample of stock_env()
    :param name: str of length 6, stock name
    :param threshold_u: int, buy if the price above the threshold
    :param threshold_d: int, sell if the price below the threshold
    :param pos_step: int, each time buy/sell 1/step of the total position
    :param mode: str, 'train' set the environment to the first half of the dataset,
                      'test' set the environment to the second half of the dataset,
                      else the whole dataset
    :param need_plot: True/False, whether or not show the picture
    :param ret_his: True/False, True to return the complete price series during the time period,
                                or False just return the final return
    '''
    res = 1
    env.mode = mode
    env.position_split = pos_step
    observation = env.reset()
    done = False
    his = [res]
    while not done:
        # print(env.step(name))
        if observation[-1] >= threshold_u:
            action = 2
        elif observation[-1] <= threshold_d:
            action = 0
        else:
            action = 1
        observation, score, done, info = env.step(action)
        # pos = 1
        # print(observation, pos)
        # print(score)
        res += score
        his.append(res)
    if need_plot:
        plt.plot(his)
        plt.show()
    if ret_his:
        return his
    return res

def greedy_thresahold_find(model_name, etf_name):
    # Using Brute Force to test each pair of (up_threshold, down_threshold) and show the
    # score of the pair of best return. Then set the threshold in config.py
    env = stock_env(model_name, etf_name, mode='all', window=1)
    best_threshold_u = 0
    best_threshold_d = 0
    best_score = 1
    best_step = 1
    for threshold_u in range(100):
        for step in range(1, 5):
            for threshold_d in range(-100, threshold_u):

                res = Monte_Carlo_Simulation(env, threshold_u / 10, threshold_d / 10, step)
                if res > best_score:
                    best_score = res
                    best_threshold_u = threshold_u / 10
                    best_step = step
                    best_threshold_d = threshold_d / 10
    print('The best threshold_u for {} is {} with step {}, threshold_d for {}, the profit is {}'.format(model_name,
                                                                                                        best_threshold_u,
                                                                                                        best_step,
                                                                                                        best_threshold_d,
                                                                                                        best_score))

def cal_weights(xlst, end_date=time.strftime('%Y-%m-%d')):
    n_stk = len(xlst)
    data_dir = zsys.rdatCNX
    # 以指数时间作为index，规范时间序
    data = pd.DataFrame()
    data['date'] = pd.read_csv(zsys.rdatCNX + '399300.csv', index_col=0).index
    data = data.set_index('date')
    data.sort_index(ascending=True, inplace=True)
    # 根据代码列表，分别读取股票数据
    for code in xlst:
        df_stk = pd.read_csv(data_dir + '/' + code + '.csv', index_col=0)
        df_stk = df_stk[df_stk.index<=end_date]
        df_stk.sort_index(ascending=True, inplace=True)
        # 求出return矩阵
        data[code] = df_stk.close.pct_change() * 100

    # 取最近三年数据计算
    data = data.tail(252 * 3)
    print(data)
    # 缺失数据补0，代表没有变化
    data.fillna(0, inplace=True)
    # 数据按列标准化
    x = data[xlst].values
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x = (x - mu) / sigma
    # 求return协方差矩阵，shape为(N*N)
    conv = np.cov(x.T)
    conv = np.matrix(conv)
    # SVD分解，求出特征值
    U, S, V = np.linalg.svd(conv)
    # 根据return矩阵大小，求出标准随机矩阵特征值范围
    Q = x.shape[0] / x.shape[1]
    lmax = 1 + (1 / Q) + 2 * np.sqrt(1 / Q)
    lmin = 1 + (1 / Q) - 2 * np.sqrt(1 / Q)
    # 将特征值中落在随机矩阵特征值中的项置为0，更新协方差矩阵
    newS = np.diag(np.where((S > lmin) & (S < lmax), 0, S))
    new_conv = U * newS * V
    # 将risk视为最小化问题，采用公式求解权重
    try:
        A = new_conv.I * np.ones((n_stk, 1))
        B = A.T * np.ones((n_stk, 1))
        weights = A / B
    except:
        A = conv.I * np.ones((n_stk, 1))
        B = A.T * np.ones((n_stk, 1))
        weights = A / B
    return dict(zip(xlst, np.array(weights).tolist()))


if __name__ == '__main__':
     #df1 = pd.read_csv('test.csv', index_col=0)
     #df1 = df1.loc['2020/8/7':'2020/12/28']
     # mkt_cmp(df1.cum_profit, start_date='2020-01-01')
     # print(cal_weights(['000016', '399300','399006']))
     env = stock_env('SVM_target30_classify5_inx-399006_loss-r2_lda_2021', '159915')
     print(Monte_Carlo_Simulation(env, threshold_u=4, threshold_d=-7, mode='all'))
