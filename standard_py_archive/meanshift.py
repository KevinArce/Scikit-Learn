import pandas as pd # data processing, CSV file I/O (excel)

from sklearn.cluster import MeanShift # MeanShift is a clustering algorithm that uses the mean shift algorithm to find clusters in a dataset.

if __name__ == "__main__": # if this file is run directly, run the following code

    dataset = pd.read_csv("data/candy.csv") # load data from candy.csv
    print(dataset.head(5)) # print first 5 rows of dataset

    X = dataset.drop(['competitorname'], axis = 1) # create X matrix from dataset

    meanshift = MeanShift().fit(X) # fit MeanShift to X matrix (clusters are found) and return fitted object (cluster centroids) and labels (cluster labels)
    print("="*80)
    print("MeanShift:", max(meanshift.labels_)) # print number of clusters found by MeanShift (max label) (max label is the number of clusters)
    print("="*80)
    print("MeanShift:", meanshift.cluster_centers_) # print cluster centers found by MeanShift (cluster centroids) (cluster centroids are the cluster centroids)

    print("="*80)
    dataset['meanshift'] = meanshift.labels_ # add cluster labels to dataset 
    print(dataset) # print dataset with cluster labels added to it
    