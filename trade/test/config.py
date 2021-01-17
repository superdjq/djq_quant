# This is the config file of main_etf_trader
# This trader combines three base model: sz50, hs300, cyb
# This trader creates daily position of each etf, and trace daily return of the portfolio
# You must define your weights of each security

weights = {'hs300': 0.2,
           'sz50': 0.45,
            'cyb': 0.35}

# You must define your model_name of each security which trained by djq_train_model


model_names = {'hs300': 'ensemble_ADA_target30_classify5_inx-399300_loss-r2_proba_2021',
            'sz50': 'ensemble_ADA_target10_classify5_inx-000016_loss-r2_proba_2021',
               'cyb': 'ensemble_ADA_target30_classify5_inx-399006_loss-r2_proba_2021'}


# model_names = {'sz50':"F:\\model\\result\\2020-08-05\\ensemble_ADA_target10_classify5_inx-2020sz50_loss-r2_proba_working\\15-36-53-ensemble_ADA_target10_classify5_inx-2020sz50_loss-r2_proba_working_result.csv",
#                 'hs300':"F:\\model\\result\\2020-08-05\\ensemble_ADA_target30_classify5_inx-hs300_loss-r2_proba_working\\15-49-33-ensemble_ADA_target30_classify5_inx-hs300_loss-r2_proba_working_result.csv",
#                 'cyb':"F:\\model\\result\\2020-08-05\\ensemble_ADA_target30_classify5_inx-cyb_loss-r2_proba_working\\15-10-47-ensemble_ADA_target30_classify5_inx-cyb_loss-r2_proba_working_result.csv"}


# You must define the threshold for your model to long or short the security

thresholds_u = {'hs300': 3.9,
           'sz50': 6.3,
            'cyb': 4.6}

thresholds_d = {'hs300': -2.4,
           'sz50': -1.2,
            'cyb': 1.8}

# You must define the percentage to buy or sell if your model shot a signal

steps = {'hs300': 4,
           'sz50': 4,
            'cyb': 4}

etf_names = {'hs300': '510300',
             'sz50': '510050',
             'cyb': '159915'}

total_cash = 75000


