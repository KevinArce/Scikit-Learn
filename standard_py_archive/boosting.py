import pandas as pd # data processing, CSV file I/O (excel)

from sklearn.ensemble import GradientBoostingClassifier # GradientBoostingClassifier is a meta-estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

from sklearn.model_selection import train_test_split # train/test split
from sklearn.metrics import accuracy_score # accuracy score 

if __name__ == "__main__": # if this file is run directly, run the following code

    dt_heart = pd.read_csv("data/heart.csv") # load data
    print(dt_heart['target'].describe()) # print target description

    X = dt_heart.drop(['target'], axis = 1) # create X matrix
    y = dt_heart['target'] # create y matrix 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42) # split data into training and testing sets with 35% test size and random state 42

    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train) # fit GradientBoostingClassifier to training data with 50 estimators 
    boost_pred = boost.predict(X_test) # predict y values using GradientBoostingClassifier
    print("="*80)
    print("Gradient Boosting Classifier Accuracy:", accuracy_score(y_test, boost_pred)) # print accuracy score of GradientBoostingClassifier 
    