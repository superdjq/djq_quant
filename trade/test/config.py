# This is the config file of main_etf_trader
# This trader combines three base model: sz50, hs300, cyb
# This trader creates daily position of each etf, and trace daily return of the portfolio
# You must define your weights of each security

weights = {'399006': 1}

# You must define your model_name of each security which trained by djq_train_model


model_names = {'399006': 'SVM_target30_classify5_inx-399006_loss-r2_modelling_2021'}


# model_names = {'sz50':"F:\\model\\result\\2020-08-05\\ensemble_ADA_target10_classify5_inx-2020sz50_loss-r2_proba_working\\15-36-53-ensemble_ADA_target10_classify5_inx-2020sz50_loss-r2_proba_working_result.csv",
#                 'hs300':"F:\\model\\result\\2020-08-05\\ensemble_ADA_target30_classify5_inx-hs300_loss-r2_proba_working\\15-49-33-ensemble_ADA_target30_classify5_inx-hs300_loss-r2_proba_working_result.csv",
#                 'cyb':"F:\\model\\result\\2020-08-05\\ensemble_ADA_target30_classify5_inx-cyb_loss-r2_proba_working\\15-10-47-ensemble_ADA_target30_classify5_inx-cyb_loss-r2_proba_working_result.csv"}


# You must define the threshold for your model to long or short the security

thresholds_u = {'399006': 5}

thresholds_d = {'399006': 1.8}

# You must define the percentage to buy or sell if your model shot a signal

steps = {'399006': 1}




