# Cross-validation is a statistical method used to estimate the skill of machine learning models in the context of their data.
# It is a way to evaluate the performance of a model by splitting the data into a training set and a test set.
# The training set is used to train the model and the test set is used to evaluate the model.
# The test set is usually a subset of the data that is not used to train the model.

import pandas as pd # data processing, CSV file I/O (excel)
import numpy as np # n-dimensional array processing (array manipulation) and linear algebra

from sklearn.tree import DecisionTreeRegressor # DecisionTreeRegressor is a decision tree regressor. It is a supervised learning algorithm that is used to make predictions based on a tree structure.

from sklearn.model_selection import ( # model selection functions (cross-validation, grid search, etc.)
cross_val_score, # cross-validation score function 
KFold) # KFold is a cross-validation generator that is used to iterate over partitions of the data used for training each successive model.

if __name__ == "__main__": # if this file is run directly, run the following code (if not, the following code will not be run)
    dataset = pd.read_csv("../data/felicidad.csv") # load data from felicidad.csv

    X = dataset.drop(['country', 'score'], axis = 1) # create X matrix from dataset (drop country and score columns) (X matrix is the dataset without country and score columns)
    y = dataset['score'] # create y matrix from dataset (y matrix is the dataset with only the score column)

    model = DecisionTreeRegressor(max_depth=5) # create DecisionTreeRegressor model with max_depth=5
    score = cross_val_score(model, X, y, cv=3,  scoring='neg_mean_squared_error') # cross-validate model with X and y and return cross-validation score (negative mean squared error) with 3 folds and scoring='neg_mean_squared_error'
    print("="*80)
    print("Mean Square Error of Predictions:", np.abs(np.mean(score))) # print mean squared error of predictions (mean squared error is the mean of the mean squared error of the predictions)

    kf = KFold(n_splits=3, shuffle=True, random_state=42) # create KFold object with 3 folds and shuffle=True and random_state=42
    for train_index, test_index in kf.split(X): # iterate over train_index and test_index from kf.split(X)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index] # create X_train and X_test from X.iloc[train_index] and X.iloc[test_index]. iloc is integer location based indexing for selection by position. 
        y_train, y_test = y.iloc[train_index], y.iloc[test_index] # create y_train and y_test from y.iloc[train_index] and y.iloc[test_index]. iloc is integer location based indexing for selection by position.
        model.fit(X_train, y_train) # fit model with X_train and y_train
        y_pred = model.predict(X_test) # predict y_pred with model and X_test
        print("="*80)
        print("Mean Square Error of Predictions:", np.abs(np.mean(y_pred - y_test))) # print mean squared error of predictions 
        print("="*80)
        

    