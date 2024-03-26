import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D', 'YR'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)
lr = LogisticRegression()
nb = GaussianNB()
da = LinearDiscriminantAnalysis()
rf = RandomForestClassifier(random_state=2022)
## w/o pass through
stack = StackingClassifier([('LR',lr),('NB',nb),('DA',da)],
                           final_estimator=rf)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = stack.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

## with pass through
stack = StackingClassifier([('LR',lr),('NB',nb),('DA',da)],
                           final_estimator=rf,
                           passthrough=True)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = stack.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

######################## Grid Search CV #####################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
stack = StackingClassifier([('LR',lr),('NB',nb),('DA',da)],
                           final_estimator=rf,
                           passthrough=True)
print(stack.get_params())
params = {'LR__C':np.linspace(0,5,5),
          'final_estimator__max_features':[2,4,6,8]}
gcv = GridSearchCV(stack, param_grid=params,
                   cv=kfold, verbose=3, scoring='roc_auc')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############### Vehicle #########################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Vehicle Silhouettes")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
vehicle = pd.read_csv("Vehicle.csv")
y = vehicle['Class']
X = vehicle.drop('Class', axis=1)

le = LabelEncoder()
le_y = le.fit_transform(y)

da = LinearDiscriminantAnalysis()
scaler = StandardScaler()
svm = SVC(probability=True, random_state=2022, kernel='linear')
pipe_svm = Pipeline([('STD',scaler),('SVML',svm)])
dtc = DecisionTreeClassifier(random_state=2022)
clf = XGBClassifier(random_state=2022)

stack = StackingClassifier([('DA',da),('SVM',pipe_svm),('TREE',dtc)],
                           final_estimator=clf,
                           passthrough=True)
print(stack.get_params())
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = {'SVM__SVML__C':np.linspace(0.001,8,10),
          'TREE__max_depth':[None,4],
          'TREE__min_samples_split':[2,4,10],
          'TREE__min_samples_leaf':[1,4],
          'final_estimator__n_estimators':[50,100],
          'final_estimator__learning_rate':[0.1,0.5],
          'final_estimator__max_depth':[3,5]}
gcv = GridSearchCV(stack, param_grid=params,
                   cv=kfold, verbose=3, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



