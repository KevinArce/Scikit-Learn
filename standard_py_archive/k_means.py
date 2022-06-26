from matplotlib import pyplot as plt # Importing matplotlib library for plotting graphs
import pandas as pd # data processing, CSV file I/O (excel) (pandas is a dataframe library)
from sklearn import datasets # datasets is a collection of datasets that can be used in machine learning algorithms

from sklearn.cluster import MiniBatchKMeans # MiniBatchKMeans is a clustering algorithm that uses the mini-batch k-means algorithm to find clusters in a dataset.
from sklearn.metrics import silhouette_score # silhouette score is a measure of how well a clustering performs on a dataset. It is calculated by comparing the average distance of each data point to its assigned cluster.
import seaborn as sns # statistical data visualization library (for plotting)

if __name__ == "__main__":

    dataset = pd.read_csv("../data/candy.csv") # load data from candy.csv  
    print("="*80)
    print(dataset.head(10)) # print first 10 rows of dataset 

    X = dataset.drop(['competitorname'], axis = 1) # create X matrix from dataset 

    # Since we're working with a very small dataset, we'll use MiniBatchKMeans with a small number of clusters (4)
    # MiniBatchKMeans is a clustering algorithm that uses the mini-batch k-means algorithm to find clusters in a dataset.
    # Also, we'll use the default parameters for the algorithm.

    # Because this is an unsupervised algorithm, we don't need to fit the model to the data or split the data into training and testing sets. 
    # We can just use the fit_predict method.
    # fit_predict returns the cluster labels for the given data. The cluster labels are stored in the labels_ attribute of the object.

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print("="*80)
    print("Total of Clusters: ", len(kmeans.cluster_centers_)) # print number of clusters found by MiniBatchKMeans (cluster centers)
    print("="*80)
    print("Prediction: ", kmeans.predict(X)) # print prediction of MiniBatchKMeans (cluster labels) (cluster labels are the cluster labels)
    print("="*80)

    # Now we can add the cluster labels to the dataset. We can do this by adding a new column to the dataset.
    dataset['kmeans'] = kmeans.predict(X) # add cluster labels to dataset
    print(dataset) # print dataset with cluster labels added to it
    print("="*80)

    # We can also use the silhouette score to evaluate the quality of the clustering.
    # The silhouette score is a measure of how well the data is clustered.
    # The silhouette score is calculated using the mean silhouette score for all the samples.

    silhouette_avg = silhouette_score(X, kmeans.labels_) # calculate silhouette score for dataset
    print("Silhouette score:", silhouette_avg) # print silhouette score for dataset
    print("="*80)
    
    # In order to visualize the clusters, we'll use Seaborn's pairplot. 
    # The pairplot function takes in a dataframe and plots the data in a grid.
    # The pairplot function also takes in a list of the names of the columns that we want to plot.
    print("Pairplot loading, please wait... ")
    sns.pairplot(dataset, hue='kmeans') # plot pairplot of dataset with cluster labels added to it
    #Let's put a delay of 5 seconds before closing the plot.
    plt.show(block=False) # show plot (block=False means that the plot will not be blocked)
    plt.pause(10) # pause for 10 seconds
    plt.close() # close plot
    print("Pairplot loaded!")