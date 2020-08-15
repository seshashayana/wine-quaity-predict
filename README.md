# wine-quaity-predict
This application predicts the wine quality from the parameters provided. The model is built on a public dataset available on public platforms.

The data is first split into train and test sets using train_test_split module from Sklearn.model_selection library

Further the parameters are scaled using StandardScaler module from SKLEARN.

For prediction of Wine Quality (3-8), I have employed several models (SKLEARN Package).
1. RandomForestRegressor
2. LinearRegression
3. LogisticRegression
4. Ridge Regression 
5. Support vector Classification

Among these models to select the best parameters, hyperparameter tuning is carried out using GridSerchCV library from SKLEARN package.

Based on the r2_scores and mean_squared_error values, LinearRegresson algorithm is used for deploying the model.
