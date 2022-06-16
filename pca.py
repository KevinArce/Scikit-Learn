import pandas as pd # data processing, CSV file I/O (excel) 
import sklearn # machine learning library 
import matplotlib.pyplot as plt # plotting library 

from sklearn.decomposition import PCA # principal component analysis 
from sklearn.decomposition import IncrementalPCA # incremental principal component analysis 

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
    
    print ("X_train shape:", X_train.shape) # print X_train shape (number of rows, number of columns)
    print ("y_train shape:", y_train.shape) # print y_train shape (number of rows, number of columns)

    # perform PCA on training data 
    # n_components = number of components to keep. If n_components is not set, it will be set to the number of features.
    pca = PCA(n_components=3) # create PCA object with 3 components 
    pca.fit(X_train) # fit PCA object to training data 

    ipca = IncrementalPCA(n_components=3, batch_size=10 ) # create IncrementalPCA object with 3 components and batch size of 10
    ipca.fit(X_train) # fit IncrementalPCA object to training data 

    # plot PCA components
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_) # plot explained variance ratio 
    plt.show() # show plot