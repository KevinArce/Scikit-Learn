import pandas as pd # data processing, CSV file I/O (excel) 
import sklearn # machine learning library

from sklearn.linear_model import LinearRegression # linear regression is a linear model that fits a linear function to the data.
from sklearn.linear_model import Lasso # lasso is a linear model that uses the L1 penalty
from sklearn.linear_model import Ridge # ridge is the algorithm that uses the Ridge regularization term to solve the linear regression problem.

from sklearn.model_selection import train_test_split # train/test split 
from sklearn.metrics import mean_squared_error # mean squared error 

if __name__ == "__main__": # if this file is run directly, run the following code 
    dataset = pd.read_csv("data/felicidad.csv") # load data 
    print(dataset.describe()) # print dataset description

    X = dataset[['gdp','family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']] # create X matrix 
    y = dataset['score'] # create y matrix

    print(X.shape) # print X matrix shape
    print(y.shape) # print y matrix shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # split data into training and testing sets with 30% test size and random state 42 
    
    # linear regression 
    modelLinear = LinearRegression().fit(X_train, y_train) # create linear regression object and fit to training data 
    y_predict_linear = modelLinear.predict(X_test) # predict y values using linear regression model 

    # lasso 
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train) # create lasso object with alpha 0.02 and fit to training data
    y_predict_lasso = modelLasso.predict(X_test) # predict y values using lasso model

    # ridge 
    modelRidge = Ridge(alpha=1).fit(X_train, y_train) # create ridge object with alpha 1 and fit to training data
    y_predict_ridge = modelRidge.predict(X_test) # predict y values using ridge model

    # print linear regression score 
    linear_loss = mean_squared_error(y_test, y_predict_linear) # calculate mean squared error of linear regression model
    print("Linear regression loss:", linear_loss) # print linear regression loss

    # print lasso score
    lasso_loss = mean_squared_error(y_test, y_predict_lasso) # calculate mean squared error of lasso model
    print("Lasso loss:", lasso_loss) # print lasso loss

    # print ridge score
    ridge_loss = mean_squared_error(y_test, y_predict_ridge) # calculate mean squared error of ridge model
    print("Ridge loss:", ridge_loss) # print ridge loss


    # Divider 
    print("="*80)

    # print linear regression coefficients
    print("Linear regression coefficients:", modelLinear.coef_) # print linear regression coefficients
    # Divider 
    print("="*80)
    
    # print lasso coefficients
    print("Lasso coefficients:", modelLasso.coef_) # print lasso coefficients
    # Divider 
    print("="*80)
    
    # print ridge coefficients
    print("Ridge coefficients:", modelRidge.coef_) # print ridge coefficients
