import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import os
from sklearn.linear_model import  ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor 

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")
concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
elastic = ElasticNet()
dtr = DecisionTreeRegressor(random_state=2022)
clf = XGBRegressor(random_state=2022)
stack = StackingRegressor([('ELASTIC',elastic),('TREE',dtr)],
                          final_estimator=clf, passthrough=True)
print(stack.get_params())
params = {'ELASTIC__alpha':np.linspace(0,10,5),
          'ELASTIC__l1_ratio':np.linspace(0,1,5),
          'TREE__max_depth':[None,3],
          'TREE__min_samples_split':[2,5,10],
          'TREE__min_samples_leaf':[1,5],
          'final_estimator__n_estimators':[50,100],
          'final_estimator__learning_rate':[0.1,0.5],
          'final_estimator__max_depth':[3,5]}

gcv = GridSearchCV(stack, param_grid=params, verbose=3, 
                   cv=kfold, scoring='r2')

gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)




