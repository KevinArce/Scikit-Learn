import pandas as pd # data processing, CSV file I/O (excel)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # KNeighborsRegressor is a k-nearest neighbors regressor.
from sklearn.ensemble import BaggingClassifier, BaggingRegressor # BaggingRegressor is a meta-estimator that fits base estimators each on random subsets of the original dataset and then aggregate their individual predictions to yield a final prediction.

from sklearn.model_selection import train_test_split # train/test split
from sklearn.metrics import accuracy_score # accuracy score 

if __name__ == "__main__": # if this file is run directly, run the following code

    dt_heart = pd.read_csv("data/heart.csv") # load data
    print(dt_heart['target'].describe()) # print target description

    X = dt_heart.drop(['target'], axis = 1) # create X matrix
    y = dt_heart['target'] # create y matrix 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42) # split data into training and testing sets with 35% test size and random state 42

    knn_class = KNeighborsClassifier().fit(X_train, y_train) # fit KNeighborsClassifier to training data
    knn_pred = knn_class.predict(X_test) # predict y values using KNeighborsClassifier
    print("="*80)
    print("KNN Classifier Accuracy:", accuracy_score(y_test, knn_pred)) # print accuracy score of KNeighborsClassifier

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50, random_state=42).fit(X_train, y_train) # fit BaggingClassifier to training data with 50 estimators and random state 42 and base estimator KNeighborsClassifier 
    bag_pred = bag_class.predict(X_test) # predict y values using BaggingClassifier
    print("="*80)
    print("Bagging Classifier Accuracy:", accuracy_score(y_test, bag_pred)) # print accuracy score of BaggingClassifier
