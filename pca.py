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