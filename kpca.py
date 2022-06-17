import pandas as pd # data processing, CSV file I/O (excel) 
import sklearn # machine learning library 
import matplotlib.pyplot as plt # plotting library 

from sklearn.decomposition import KernelPCA # kernel principal component analysis 

from sklearn.linear_model import LogisticRegression # logistic regression 

from sklearn.preprocessing import StandardScaler # standard scaler 

from sklearn.model_selection import train_test_split # train/test split 

if __name__ == "__main__":
    # load data 
    dt_heart = pd.read_csv("data/heart.csv") 

    print(dt_heart.head(5)) # print first 5 rows

    dt_features = dt_heart.drop(["target"], axis=1) # drop target column 
    dt_target = dt_heart["target"] # target column

    # standardize data
    dt_features = StandardScaler().fit_transform(dt_features) # fit and transform to standardize data

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42) # split data into training and testing sets with 30% test size and random state 42 
    
    kpca = KernelPCA(n_components=4, kernel='poly', gamma=0.1) # create KernelPCA object with 4 components and polynomial kernel with gamma 0.1. Gamma is the kernel parameter for the polynomial kernel.
    kpca.fit(X_train) # fit KernelPCA object to training data

    dt_train = kpca.transform(X_train) # transform training data using KernelPCA
    dt_test = kpca.transform(X_test) # transform testing data using KernelPCA

    logistic = LogisticRegression(solver='lbfgs') # create logistic regression object with solver lbfgs

    logistic.fit(dt_train, y_train) # fit logistic regression model to training data
    print("(KPCA) Logistic regression score:", logistic.score(dt_test, y_test)) # print logistic regression score