# djq_quant

China stock predictor and trade manager using ensemble machine learning algorithm.
## How to use
You can simply create your model with a single line project name, like
- SVM_target30_classify5_inx-399006_loss-r2_modelling_2021
- ensemble_ADA_target30_classify5_inx-399006_loss-r2_proba_working_2021
### Project name components
- machine learning method, like "SVM", "RF" for random-forest, "ET" for extra-tree, 
  Start with "emsemble" means your project consists of several basic classifiers, and "ADA" for using adaboost.
- key "target"+"n" means you want to predict the change after n days
- key "classify"+"n" means how many classes you want to qut your train sets by using "pandas.qcut()"
- key "inx" means your model is based on the constituents of the index, you can also define your own stock portfolio
- key "loss" means the loss function you want to use, like built-in func "R2" and "f1" and other customized functions
- add key "proba" means the classifiers give you probabilities of each class
- key "modelling" divide your dataset into train-set and test-set, and give you the result on test-set
      "working" use the newest data to train the model
- other labels for differentiation
## Process flow
![Image text](https://raw.githubusercontent.com/superdjq/djq_quant/master/model%20flowchart.png)
## Data source & data layout
You can use your local data set or download china stock data by using 
```python 
djq_data_processor.stock_update('D')
djq_data_processor.index_update('15')
```
with "D" for daily data and "n" for every n minutes data, which n in {5,15,30,60}
### folder structure
```
main_folder 
├── data
│   ├── day
│   │    ├── stk
│   │    └── inx
│   └── min               
│        ├── stk
│        └── inx
├── model
│     ├── book         // each classifier has a book to record best params and .pkl file location
│     ├── result       // record predict results for each day 
│     └── model_folder                  
├── trade
      └── trade_folder     
```  
## Demo
### create a model
```python 
pj = djq_train_model.StcokClassifier('RF_target10_classify5_inx-399300_loss-profit_working')
pj.train() // train model with local data
df = pj.daily_predict() 
```
### customize algorithm
```python 
Add your algorithm in dict BASE_MODELS
BASE_MODELS = {'ABC': ABC(class_weight='balanced', probability=True)}
ADD your algorithm's optimal params in dict BASE_MODEL_PARAMS
BASE_MODEL_PARAMS = {'ABC': {'C': [1, 10, 100, 1000, 10000]}}
```
### create a trade manager
you may create a folder under folder "trade", and create a config file in the folder, using config.py as a sample
```python 
trade = Trader('etf_trade_manager')
trade.daily_monitor()
```
## Todo list
- [ ] Add comments
- [ ] Utils module
- [ ] Data base support
- [ ] Automatic trading module
- [ ] Thresholds finding by RL
- [ ] Diversified indicators
