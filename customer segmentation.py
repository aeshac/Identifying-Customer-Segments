# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import libraries necessary for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vpython import *
import vpython as vp

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print ("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?")
    
# Display a description of the dataset
stats = data.describe()
stats

# Select three indices of to sample from the dataset
indices = [26,176,392]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print ("Chosen samples of wholesale customers dataset:")
display(samples)

# Feature Relevance

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# drop the feature
new_data = data.drop(['Milk'],axis=1)

# split the data
X_train, X_test, y_train, y_test = train_test_split(new_data,data['Milk'],test_size=0.25,random_state=101)

# Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=101).fit(X_train,y_train)

# Report the score of the prediction using the testing set
score = regressor.score(X_test,y_test)

print (score)

Fresh_data = data.drop(['Fresh'],axis=1)

# split the data
X_train, X_test, y_train, y_test = train_test_split(Fresh_data,data['Fresh'],test_size=0.25,random_state=101)

# Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=101).fit(X_train,y_train)

# Report the score of the prediction using the testing set
score = regressor.score(X_test,y_test)

print (score)

Detergents= data.drop(['Detergents_Paper'],axis=1)

# split the data
X_train, X_test, y_train, y_test = train_test_split(Detergents,data['Detergents_Paper'],test_size=0.25,random_state=101)

# Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=101).fit(X_train,y_train)

# Report the score of the prediction using the testing set
score = regressor.score(X_test,y_test)

print (score)

Grocery_data = data.drop(['Grocery'],axis=1)

# split the data
X_train, X_test, y_train, y_test = train_test_split(Grocery_data,data['Grocery'],test_size=0.25,random_state=101)

# Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=101).fit(X_train,y_train)

# Report the score of the prediction using the testing set
score = regressor.score(X_test,y_test)

print (score)


# Correlation plot

from pandas.plotting import scatter_matrix
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# Scale the data using the natural logarithm
log_data = data.apply(lambda x: np.log(x))

# Scale the sample data using the natural logarithm
log_samples = samples.apply(lambda x: np.log(x))

# Produce a scatter matrix for each pair of newly-transformed features
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

display(log_samples)


outliers  = []

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1) * 1.5
    
    # Display the outliers
    print ("Data points considered outliers for the feature '{}':".format(feature))
    out = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(out)
    outliers = outliers + list(out.index.values)
    

#Creating list of more outliers which are the same for multiple features.
outliers = list(set([x for x in outliers if outliers.count(x) > 1]))    

print ("Outliers: {}".format(outliers))

# Remove the outliers, if any were specified 
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
print ("The good dataset now has {} observations after removing outliers.".format(len(good_data)))

# PCA

from sklearn.preprocessing import StandardScaler
pc = StandardScaler().fit_transform(good_data) # normalizing the features
pc.shape

# check if normalized data has mean of zero and standard deviation of one

np.mean(pc)
np.std(pc)

feat_cols = ['feature'+str(i) for i in range(pc.shape[1])]
normalisedPC = pd.DataFrame(pc,columns=feat_cols)
normalisedPC.tail()


from sklearn.decomposition import PCA
pcac = PCA().fit(good_data)
pca_samples = pca.transform(log_samples)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(pc)

pca_df = pd.DataFrame(data = principalComponents,columns = ['principal component 1', 'principal component 2'])
pca_df.tail

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

reduced_data = pca.transform(good_data)


# Clustering

n_clusters = [8,6,4,3,2]

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

for n in n_clusters:
    
    # Apply your clustering algorithm to the good data 
    clusterer = GaussianMixture(n_components=n).fit(reduced_data)

    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)
    print(preds)
    
    # Find the cluster centers
    centers = clusterer.means_

    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data,preds)
    
    print ("The silhouette_score for {} clusters is {}".format(n,score))
    
# The silhouette_score for 8 clusters is 0.1324394822359856
# The silhouette_score for 6 clusters is 0.13372603930207078
# The silhouette_score for 4 clusters is 0.17979816713423127
# The silhouette_score for 3 clusters is 0.193221674196826
# The silhouette_score for 2 clusters is 0.2917401885735662
    
# We see that the silhouette_score for 2 clusters is the best 

# Data Recovery

# Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

# Display the predictions
for i, pred in enumerate(preds):
    print ("Sample point", i, "predicted to be in Cluster", pred)
