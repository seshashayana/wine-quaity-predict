import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv("winequality-red.csv", sep = ";")

X = dataset.drop('quality', axis=1)
y = dataset.quality

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

pipeline.get_params()

hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth' : [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit and tune model
clf.fit(X_train, y_train)

clf.refit

y_pred = clf.predict(X_test)

print("Random Forest Model Scores")
print("Mean Squared Error: ",mean_squared_error(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred),"/n")

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


pipeline_logreg = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(max_iter=1000))
params_logreg = pipeline_logreg.get_params()

hyperparams_logreg = {'logisticregression__penalty':['l1', 'l2', 'elasticnet', 'none'],
                      'logisticregression__solver':['newton-cg', 'lbfgs', 'liblinear']}

clf_logreg = GridSearchCV(pipeline_logreg, hyperparams_logreg, cv=10);

# Fit and tune model
clf_logreg.fit(X_train, y_train);

clf_logreg.refit

y_pred_logreg = clf_logreg.predict(X_test)
print("Logistic Regression Model Scores")
print("Mean Squared Error: ",mean_squared_error(y_test, y_pred_logreg))
print("R2 Score", r2_score(y_test, y_pred_logreg),"/n")

clf_linreg = LinearRegression()
clf_linreg_fit = clf_linreg.fit(X_train, y_train)

y_pred_linreg = clf_linreg_fit.predict(X_test)
print("Linear Regression Model Scores")
print("Mean Squared Error: ",mean_squared_error(y_test, y_pred_linreg))
print("R2 Score", r2_score(y_test, y_pred_linreg),"/n")

from sklearn.svm import SVC

pipeline_svc = make_pipeline(preprocessing.StandardScaler(), SVC())
params_svc = pipeline_svc.get_params()

hyperparams_svc = {'svc__kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                      'svc__gamma':['scale', 'auto']}

clf_svc = GridSearchCV(pipeline_svc, hyperparams_svc, cv=10);

# Fit and tune model
clf_svc.fit(X_train, y_train);

clf_svc.refit

y_pred_svc = clf_svc.predict(X_test)
print("Support Vector Classification Model Scores")
print("Mean Squared Error: ",mean_squared_error(y_test, y_pred_svc))
print("R2 Score", r2_score(y_test, y_pred_svc),"/n")

from sklearn.linear_model import Ridge

clf_ridge = Ridge()
clf_ridge_fit = clf_ridge.fit(X_train, y_train)

y_pred_ridge = clf_ridge_fit.predict(X_test)
print("Ridge Regression Model Scores")
print("Mean Squared Error: ",mean_squared_error(y_test, y_pred_ridge))
print("R2 Score", r2_score(y_test, y_pred_ridge),"/n")

import pickle

pickle.dump(clf_linreg, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
