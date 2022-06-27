import pandas as pd # data processing, CSV file I/O (excel)

from sklearn.model_selection import RandomizedSearchCV # RandomizedSearchCV is a meta estimator that implements a "fit" and a "score" method.
from sklearn.ensemble import RandomForestRegressor # RandomForestRegressor is a random forest regressor. It is a meta estimator that fits a number of decision tree regressors at once.

if __name__ == "__main__":

    dataset = pd.read_csv("../data/felicidad.csv") # load data from felicidad.csv

    print(dataset) # print dataset
    print("="*80)

    X = dataset.drop(['country', 'rank', 'score'], axis = 1) # create X matrix from dataset (drop country, rank and score columns) (X matrix is the dataset without country, rank and score columns)
    y = dataset['score'] # create y matrix from dataset (y matrix is the dataset with only the score column)

    reg = RandomForestRegressor(n_estimators=100) # create RandomForestRegressor model with n_estimators=100

    parameters = { 
    
        'n_estimators': range(4,16), # create parameters dictionary with n_estimators from 4 to 16
        'criterion': ['squared_error', 'absolute_error'], # create parameters dictionary with criterion from squared_error and absolute_error 
        'max_depth': range(2,11) # create parameters dictionary with max_depth from 2 to 11
        }

    random_estimator = RandomizedSearchCV( # create RandomizedSearchCV object with reg and parameters
    reg, # reg is the regressor to use in the RandomizedSearchCV object
    parameters, # parameters is the parameters to use in the RandomizedSearchCV object
    n_iter=10 , # n_iter is the number of randomized parameter combinations to try (10)
    cv=3, # cv is the number of folds to use for cross-validation (3)
    scoring='neg_mean_absolute_error' # scoring is the scoring metric to use for cross-validation (negative mean absolute error)
    ).fit(X, y) # fit model with X and y using RandomizedSearchCV object (RandomizedSearchCV is a meta estimator that implements a "fit" and a "score" method.

    print("="*80)
    print("Best Parameters:", random_estimator.best_params_) # print best parameters from RandomizedSearchCV object
    print("="*80)
    print("Best Estimator:", random_estimator.best_estimator_) # print best estimator  from RandomizedSearchCV object (RandomForestRegressor) (RandomForestRegressor is a random forest regressor. It is a meta estimator that fits a number of decision tree regressors at once.)
    print("="*80)
    print("Best Score:", random_estimator.best_score_) # print best score from RandomizedSearchCV object (negative mean absolute error) (best score is the lowest score)
    print("="*80)
    print("Best Index:", random_estimator.best_index_) # print best index (index of the best parameter combination) 
    print("="*80)
    print("Scoring Function:", random_estimator.scorer_) # print scoring function (negative mean absolute error) (scorer_ is a function that scores the performance of an estimator)
    print("="*80)
    print("Best Prediction:", random_estimator.predict(X.iloc[[0]])) # print best prediction (prediction of the best parameter combination) (X.iloc[[0]] is the first row of X matrix) (X.iloc[[0]] is a pandas dataframe) 
    print("="*80)