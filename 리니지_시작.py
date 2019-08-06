from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import pandas as pd
import xgboost as xgb
import sklearn
# ==============================================================
data = pd.read_csv('C:/Users/CPB06GameN/Desktop/빅콘/2019_빅콘테스트_챔피언스리그_데이터/train_activity.csv')
data = data['private_shop']
data2 = pd.read_csv('C:/Users/CPB06GameN/Desktop/빅콘/2019_빅콘테스트_챔피언스리그_데이터/test1_activity.csv')
data2 = data2['private_shop']
dtrain = xgb.DMatrix(data)
dtest = xgb.DMatrix(data2)

param = {'max_depth':4, 'eta':1,'silent':1, 'objective':'binart:logistic'}
num_round = 100
bst = xgb.train(param, dtrain, num_round)

preds = bst.predict(dtest)
print(preds)

# =======================================================================
dataset = pd.read_csv('C:/Users/CPB06GameN/Desktop/빅콘/2019_빅콘테스트_챔피언스리그_데이터/train_activity.csv')
dataset.columns
x = dataset[:]
y = dataset[:]
model = XGBClassifier()
model.fit(x,y)
plot_importance(model)
pyplot.show()

from sklearn.tree import DecisionTreeClassifier


# ==================================도전=================================================
import numpy as np
import pandas as pd


from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import train_test_split



df = pd.read_csv('C:/Users/CPB06GameN/Desktop/빅콘/2019_빅콘테스트_챔피언스리그_데이터/train_activity.csv')
df['private_shop']
df.info()
training_fields = ['playtime','npc_kill','solo_exp','party_exp']
X_train, X_test, y_train, y_test = train_test_split(df[training_fields], df['day','acc_id','char_id'], test_size=0.2, random_state=42)


optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), cv_params, scoring='accuracy', cv=5, n_jobs=-1)
optimized_GBM.fit(X_train[training_fields], y_train)

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

cv_params = {}
cv_params['max_depth'] = [3,5,7]
cv_params['min_child_weight'] = [1,3,5]

ind_params = {}
ind_params['learning_rate'] = 0.1
ind_params['n_estimators'] = 1000
ind_params['seed'] = 0
ind_params['subsample'] = 0.8
ind_params['colsample_bytree'] = 0.8
ind_params['objective'] = 'binary:logistic'

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 7

optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), cv_params, scoring='accuracy', cv=5, n_jobs=-1)
optimized_GBM.fit(X_train[training_fields], y_train)

best_parameters, score, _ = max(optimized_GBM.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

test_probs = clf.predict_proba(y_train)[:,1]

d_train = xgb.DMatrix(X_train[training_fields], label=y_train)
d_valid = xgb.DMatrix(X_test[training_fields], label=y_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 200, watchlist, early_stopping_rounds=50, verbose_eval=10)


bst.save_model('리니지_첫도전_xgboost.model')
xgb.plot_importance(bst)