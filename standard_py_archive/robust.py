import pandas as pd # data processing, CSV file I/O (excel)

from sklearn.linear_model import ( # linear regression is a linear model that fits a linear function to the data.
RANSACRegressor, # RANSACRegressor is a robust estimator that is resistant to outliers. 
HuberRegressor, # HuberRegressor is a robust estimator that is resistant to outliers.
) 
from sklearn.model_selection import train_test_split # train/test split 
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR # mean squared error

if __name__ == "__main__": # if this file is run directly, run the following code 
    dataset = pd.read_csv("data/felicidad_corrupt.csv") # load data 
    print(dataset.head(5)) # print first 5 rows of dataset 

    X = dataset.drop(['country', 'score'], axis = 1) # create X matrix 
    y = dataset['score'] # create y matrix

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # split data into training and testing sets with 30% test size and random state 42

    estimadors = { # create dictionary of estimators
        'SVR': SVR(gamma= 'auto', C=1, epsilon=0.1), # SVR is a support vector regression model. Gamma is the kernel coefficient for the RBF kernel. C is the penalty parameter for the error term. Epsilon is the epsilon parameter for the epsilon-insensitive loss function.
        'RANSAC': RANSACRegressor(), # RANSACRegressor is a robust estimator that is resistant to outliers.
        'Huber': HuberRegressor(epsilon=1.35) # HuberRegressor is a robust estimator that is resistant to outliers. epsilon is the epsilon parameter for the Huber loss function.
    } # end estimators

    for name, estimador in estimadors.items(): # for each estimator in estimadors dictionary 
        estimador.fit(X_train, y_train) # fit estimator to training data 
        y_predict = estimador.predict(X_test) # predict y values using estimator 

        print(80*"=") # print divider
        print(name) # print estimator name
        print("MSE:", mean_squared_error(y_test, y_predict)) # print mean squared error of estimator 