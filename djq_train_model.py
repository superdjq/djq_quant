"""
The module 'djq_train_model' provides a streamline method to train a stock predictor model
using basic machine learning algorithm.
Define your model params all in a single project name, like
'ensemble_ADA_target30_classify5_inx-399006_loss-r2_proba_working_2021'
Project name structure: 'keyword1 + param1' + 'keyword2 + param2' + '_' + ...
the basic keywords:
- machine learning method, like "SVM", "RF" for random-forest, "ET" for extra-tree,
  Start with "ensemble" means your project consists of several basic classifiers, and "ADA" for using adaboost.
- key "target"+"n" means you want to predict the change after n days
- key "classify"+"n" means how many classes you want to qut your train sets by using "pandas.qcut()"
- key "inx" means your model is based on the constituents of the index, you can also define your own stock portfolio
- key "loss" means the loss function you want to use, like built-in func "R2" and "f1" and other customized functions
- add key "proba" means the classifiers give you probabilities of each class
- key "xlst" means the indicators you want to use in your model, several indicator combinations are defined in
        module "zsys", you can alse define your own indicator combination in "zsys"
- key "date" means the start and end year you want to used in your model, like "date2012-2020"
- key "minprofit" helps to remove stocks with max upward change below your desired value
- key "pca" set the dimension scale to reduce raw data set, default with 50
- other labels for differentiation
"""
import os, time
import pandas as pd
import numpy as np
import zsys
from djq_data_processor import get_label
from djq_talib import get_all_finanical_indicators
import tushare as ts
import dtshare as dt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed


class StcokClassifier(object):
    # BASE_DIR = 'E:\\WORK\\quant\\model\\'
    BASE_DIR = 'model/'
    # Add your algorithm in BASE_MODELS
    BASE_MODELS = {'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True),
                    'RF': RandomForestClassifier(criterion='gini',
                            max_features='auto', n_jobs=8, min_samples_split=2,
                            bootstrap=True, oob_score=False, min_samples_leaf=1,
                            random_state=None, verbose=0),
                   'ADA': AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=2000),
                                             n_estimators=100, algorithm='SAMME'),
                   'LR': LogisticRegression(penalty='l2'),
                   'ET': ExtraTreesClassifier(criterion='gini',
                            max_features='auto', n_jobs=8,
                            bootstrap=True, oob_score=False,
                            random_state=None, verbose=0)}
    # ADD your algorithm's optimal params in dict BASE_MODEL_PARAMS
    BASE_MODEL_PARAMS = {'SVM': {'C': [1, 10, 100, 1000, 10000], 'gamma': [0.01, 0.001, 0.0001]},
                        'RF': {'n_estimators': [200, 1000, 2000, 5000], 'max_depth': [10, 50, None]},
                         'ET': {'n_estimators': [200, 1000, 2000, 5000], 'max_depth': [10, 50, None],
                                'min_samples_split':[2, 10], 'min_samples_leaf': [1, 10]}}

    def __init__(self, pjNam):
        self.pjNam = pjNam
        self.ensemble = False
        self.subclassifiers = None
        self.target_day = 30
        self.classify = 5
        self.inx = 'hs300'
        self.model_type = None
        self.loss_type = None
        self.data_params = []
        self.proba = False
        self.book = pd.DataFrame(columns=('code', 'best_train_score', 'best_params', 'model_dir', 'test_profit'))
        self._parse_projectName()

    def _parse_projectName(self):
        for part in self.pjNam.split('_'):
            if part == 'ensemble':
                self.ensemble = True
                self.subclassifiers = {}
                with open(StcokClassifier.BASE_DIR + 'book/' + self.pjNam + '_classifiers.txt', 'r') as f:
                    for line in f.readlines():
                        if not line.startswith('#') and '_' in line:
                            sub_pj = line.replace('\n', '')
                            self.subclassifiers[sub_pj] = StcokClassifier(sub_pj)
            elif part.startswith('target'):
                self.target_day = int(part[6:])
            elif part.startswith('classify'):
                #self.classify = {5: (int(part[8:]), -1, 2),
                #                 15: (int(part[8:]), -4, 8),
                #                 30: (int(part[8:]), -7, 14),
                #                 60: (int(part[8:]), -8, 16)}[self.target_day]
                self.classify = int(part[8:])
            elif part in StcokClassifier.BASE_MODELS:
                self.model_type = part
            elif part.startswith('loss'):
                self.loss_type = {'profit': profit_score,
                                  'f1': 'f1_weighted',
                                  'r2': 'r2'}[part[5:]]
            elif part.startswith('inx'):
                self.inx = part[4:]
            elif part.startswith('xlst'):
                xlst_name = part[5:].split('+')
                xlst = []
                for name in xlst_name:
                    xlst += {'ohlcV': zsys.ohlcVLst, 'chart': zsys.stcokcharts_indicators,
                             'last': zsys.last_data, 'all': zsys.TDS_talib_indicators_all,
                             'ohlc': zsys.ohlcLst, 'talib5': zsys.TDS_talib_multi_indicators_5,
                             'talib15': zsys.TDS_talib_multi_indicators_15,
                             'talib30': zsys.TDS_talib_multi_indicators_30,
                             'talibstatic': zsys.TDS_talib_multi_indicators_Static}[name]
                self.xlst = xlst
            elif part == 'proba':
                self.proba = True
            else:
                self.data_params.append(part)
        if not self.model_type:
            raise ValueError('You should clarify the TYPE of the Classifier!')


        if os.path.isfile(StcokClassifier.BASE_DIR + 'book/' + self.pjNam + '_book.csv'):
            self.book = pd.read_csv(StcokClassifier.BASE_DIR + 'book/' + self.pjNam + '_book.csv',
                                    dtype={'code': str}, encoding='GBK')
            self.mlst = list(self.book['code'].values)
        else:
            if self.ensemble:
                score = pd.DataFrame()
                for name, classifier in self.subclassifiers.items():
                    score[name] = classifier.book['best_train_score'].rank(ascending=False)
                score.index = classifier.book.code
                score['total'] = score.sum(axis=1)
                score = score.sort_values('total')
                self.mlst = list(score.index)[:300]
                score.to_csv('tmp/rank.csv')
            else:
                # finx = 'F://Dat/data/stk_' + self.inx + '.csv'
                # inx_df = pd.read_csv(finx, dtype={'code': str}, encoding='GBK')
                # self.mlst = list(inx_df['code'])
                try:
                    inx_df = dt.index_stock_cons(self.inx)
                except:
                    raise ValueError('Can not find the constitutions of index {}'.format(self.inx))
                self.mlst = list(inx_df['品种代码'])

        if not os.path.isdir(self.BASE_DIR + self.pjNam):
            os.makedirs(self.BASE_DIR + self.pjNam)



    def data_prepare(self, code, drop=True, real_time=True):
        """
        :param code: China stock code with 6 numbers, using local data set
        :param drop: set True to drop the data of days with missing information
        :param real_time: add real time data as the newest data when the market is open
        :return: train_data, train_label, test_data, test_label, df_train, df_test, class threshold
        """
        xlst = zsys.ohlcVLst + zsys.stcokcharts_indicators + zsys.last_data  # + zsys.TDS_talib_indicators_all
        train_start = '2012-01-01'
        train_end = '2019-12-31'
        n_pca = 50
        min_profit_ratio = self.target_day // 3
        for part in self.data_params:
            if part.startswith('xlst'):
                xlst_name = part[5:].split('+')
                xlst = []
                for name in xlst_name:
                    xlst += {'ohlcV': zsys.ohlcVLst, 'chart': zsys.stcokcharts_indicators,
                             'last': zsys.last_data, 'all': zsys.TDS_talib_indicators_all,
                             'ohlc': zsys.ohlcLst, 'talib5': zsys.TDS_talib_multi_indicators_5,
                             'talib15': zsys.TDS_talib_multi_indicators_15,
                             'talib30': zsys.TDS_talib_multi_indicators_30,
                             'talibstatic': zsys.TDS_talib_multi_indicators_Static}[name]
            elif part.startswith('date'):
                start, end = part[5:].split('-')
                train_start = start + '-01-01'
                train_end = end + '-12-31'
            elif part.startswith('minprofit'):
                min_profit_ratio = int(part[10:])
            elif part.startswith('pca'):
                if part[4:].isnumeric():
                    n_pca = int(part[4:])
                elif part[4:] == 'None':
                    n_pca = None

        # 数据准备
        try:
            df = pd.read_csv(zsys.rdatCN + code + '.csv')
        except:
            raise ValueError('Cannot find the file!')
        if real_time and df['date'][0] != time.strftime('%Y-%m-%d'):
            open_time = time.strptime(time.strftime('%Y-%m-%d') + ' 09:30:00', '%Y-%m-%d %H:%M:%S')
            now = time.strptime(time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
            diff = min(max(0, time.mktime(now) - time.mktime(open_time)), 2 * 60 * 60)
            open_time = time.strptime(time.strftime('%Y-%m-%d') + ' 13:00:00', '%Y-%m-%d %H:%M:%S')
            diff += min(max(0, time.mktime(now) - time.mktime(open_time)), 2 * 60 * 60)
            new = ts.get_realtime_quotes([code])
            if diff and float(new['open'][0]):
                time_multiple = 4 * 60 * 60 / diff
                line = pd.Series(dict(zip(['date', 'open', 'high', 'low', 'close', 'volume'],
                                      [time.strftime('%Y-%m-%d'), float(new['open'][0]), float(new['high'][0]),
                                       float(new['low'][0]), float(new['price'][0]), time_multiple * float(new['volume'][0][:-2])])))
                df = df.append(line, ignore_index=True)
                df = df.sort_values('date', ascending=False)
                df = df.reset_index(drop=True)
        if df.shape[0] < 252:
            raise ValueError('Not enough train data!')
        df = get_all_finanical_indicators(df)
        get_label(df, target_day=self.target_day)
        df = df[df.date >= train_start]
        if drop:
            df = df.dropna()
        #transfer_label_to_classification(df, classify=self.classify)
        if df.shape[0] < 252:
            raise ValueError('Not enough train data!')
        df['y_pct_change'] = df['y'].copy()
        df_train = df[df.date <= train_end].copy()
        df_test = df[df.date > train_end].copy()
        split, thresholds = pd.qcut(df_train['y'], self.classify, labels=range(self.classify), retbins=True)
        if thresholds[-2] < min_profit_ratio:
            pass
            # raise ValueError('Too small profit!')
        df_train['y'] = np.array(split)
        df_test['y'] = np.array(pd.cut(df_test['y'], thresholds, labels=range(self.classify)))

        # 数据清洗
        std = preprocessing.StandardScaler()
        if n_pca:
            pca_50 = PCA(n_components=n_pca)

        x_train = df_train[xlst].values
        x_train = std.fit_transform(x_train)
        if n_pca:
            x_train = pca_50.fit_transform(x_train)
        # 测试集与训练集采用相同方法处理
        x_test = df_test[xlst].values
        if x_test.shape[0]:
            x_test = std.transform(x_test)
            if n_pca:
                x_test = pca_50.transform(x_test)
        return self.subclassifiers_transfer(code, x_train), df_train['y'].values, self.subclassifiers_transfer(code, x_test), df_test['y'].values, df_train, df_test, thresholds


    def train(self):
        """
        Train every single stock in the index
        Use grid search to find the best parameters
        Create a book in path '/model/book' to record the params, model pkl file address, test result
        :return: None
        """
        for code in self.mlst.copy():
            file_path = StcokClassifier.BASE_DIR + self.pjNam + '/' + code + '.pkl'
            if os.path.isfile(file_path):
                print('Model for code:{} is already existed!'.format(code))
                continue
            print('Training code:{} for project:{}'.format(code, self.pjNam))
            try:
                x_train, y_train, x_test, y_test, df_train, df_test, thresholds = self.data_prepare(code, real_time=False)
            except ValueError as e:
                print(e.args)
                self.mlst.remove(code)
                continue
            if self.model_type in StcokClassifier.BASE_MODEL_PARAMS:
                grid = GridSearchCV(StcokClassifier.BASE_MODELS[self.model_type],
                                    param_grid=StcokClassifier.BASE_MODEL_PARAMS[self.model_type], cv=10,
                                    scoring=self.loss_type, n_jobs=8)
                grid.fit(x_train, y_train)
                model = grid.best_estimator_
                score = grid.best_score_
                params = grid.best_params_
            else:
                model = StcokClassifier.BASE_MODELS[self.model_type]
                model.fit(x_train, y_train)
                score = model.score(x_train, y_train)
                params = {}

            param_str = ''
            for p, v in params.items():
                param_str += '_' + p + str(v)
            file_path = self.pjNam + '/' + code + '.pkl'
            joblib.dump(model, filename=StcokClassifier.BASE_DIR + file_path)
            result_dict = {}
            for i in range(5):
                result_dict['tier'+str(i)] = (thresholds[i] + thresholds[i+1]) / 2
            result_dict.update({'code': code, 'best_train_score': score, 'best_params': str(params), 'model_dir': file_path})
            if len(x_test):
                y_pred = model.predict(x_test)
                profit = df_test['y_pct_change'].values[y_pred == self.classify-1].mean()
                print('The best model trained for {} get profit {} on the test set.'.format(code, profit))
                result_dict['test_profit'] = profit
                result_series = pd.Series(result_dict)
            else:
                result_series = pd.Series(result_dict)
            self.book = self.book.append(result_series, ignore_index=True)

        self.book.to_csv(StcokClassifier.BASE_DIR + 'book/' + self.pjNam + '_book.csv')
        self.mlst = list(self.book['code'].values)



    def subclassifiers_transfer(self, code, X):
        if self.ensemble and X.shape[0]:
            if self.proba:
                return np.concatenate([classifier.code_predict(code, X, proba=self.proba) for classifier in self.subclassifiers.values()], axis=1)
            predict = []
            for name, classifier in self.subclassifiers.items():
                predict.append(classifier.code_predict(code, X))
            X = np.array(predict).T
        return X

    def code_predict(self, code, X, proba=False):
        if not code in self.mlst:
            raise ValueError('the code:{} have not trained in the project:{}!'.format(code, self.pjNam))
        X = self.subclassifiers_transfer(code, X)
        model_name = self.book[self.book.code==code]['model_dir'].values[0]
        model = joblib.load(StcokClassifier.BASE_DIR + model_name)
        if proba: return model.predict_proba(X)
        return model.predict(X)

    def model_predict(self, code, train_result, real_time=True):
        '''
        :param code: string of length 6, code of china stock
        :param train_result: set True to show the result on train set, instead of test set result
        :param real_time: set True to crawl the real time data when market is open
        :return: pandas.Series, estimated change
        '''
        try:
            if train_result:
                x_test, _, _, _, df_test, _, _ = self.data_prepare(code, drop=False, real_time=real_time)
            else:
                _, _, x_test, _, _, df_test, _ = self.data_prepare(code, drop=False, real_time=real_time)
        except ValueError as e:
            print(e.args)
        model_name = self.book[self.book.code == code]['model_dir'].values[0]
        model = joblib.load(StcokClassifier.BASE_DIR + model_name)
        df_test.index = df_test.date
        df_test[code] = model.predict(x_test)
        return df_test[code]

    def daily_predict(self, train_result=False):
        today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        real_time = True
        folder = StcokClassifier.BASE_DIR + 'result/' + today + '/' + self.pjNam
        if not os.path.isdir(folder):
            real_time = False
            os.makedirs(folder)
        file_path = folder + '/' + time.strftime('%H-%M-%S-') + self.pjNam + '_result.csv'
        # file_path = folder + '/' + self.pjNam + '_result.csv'
        if not os.path.isfile(file_path):
            result = pd.concat(Parallel(n_jobs=-1)(delayed(self.model_predict)(code, train_result, real_time=real_time) for code in self.mlst), axis=1)
            result = result.sort_index()
            result = result.fillna('2')
            result = result.astype('int')
            result.to_csv(file_path)
        return pd.read_csv(file_path, index_col=0)





def profit_score(clf, X, y_true):
    '''
    sample of customized loss function
    return the mean change when predict is right
    :param clf:
    :param X:
    :param y_true:
    :return:
    '''
    y_pred = clf.predict(X)
    threshold = y_true.max()
    return y_true[y_pred==threshold].mean()

def fold_score(clf, x_train, y_train):
    y_valid = []
    batch_size = (x_train.shape[0] // 10) + 1
    for i in range(10):
        x = np.vstack((x_train[:i*batch_size], x_train[(i+1)*batch_size:]))
        y = list(y_train[:i*batch_size])+list(y_train[(i+1)*batch_size:])
        x_test = x_train[i*batch_size:(i+1)*batch_size]
        std = preprocessing.StandardScaler()
        std.fit(x)
        x = std.transform(x)
        x_test = std.transform(x_test)
        pca_50 = PCA(n_components=50)
        pca_50.fit(x)
        x = pca_50.transform(x)
        x_test = pca_50.transform(x_test)
        clf.fit(x, y)
        y_valid += list(clf.predict(x_test))
    return accuracy_score(y_pred=np.array(y_valid), y_true=y_train), r2_score(y_pred=np.array(y_valid), y_true=y_train)

if __name__ == '__main__':
    # pj = StcokClassifier('RF_target10_classify5_inx-2020sz50_loss-profit_working')
    # pj.train()
    # print('I change some file!')
    for pjNam in [
                 #'RF_target30_classify5_inx-cyb_loss-r2_working_2021',
                 # 'RF_target30_classify5_inx-cyb_loss-f1_working_2021',
                 # 'RF_target30_classify5_inx-cyb_loss-profit_working_2021',
                 #'ET_target30_classify5_inx-cyb_loss-r2_working_2021',
                 #'ET_target30_classify5_inx-cyb_loss-f1_working_2021',
                 #'ET_target30_classify5_inx-cyb_loss-profit_working_2021',
                 'SVM_target30_classify5_inx-399006_loss-r2_2021',
                 #'SVM_target30_classify5_inx-cyb_loss-f1_working_2021',
                 #'SVM_target30_classify5_inx-cyb_loss-profit_working_2021',
                 #'ensemble_ADA_target30_classify5_inx-cyb_loss-r2_proba_working_2021'
                  ]:
        pj = StcokClassifier(pjNam)
        #pj.train()
        pj.daily_predict()