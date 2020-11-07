# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 11:33:19 2020

@author: Yash
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('PCA_practice_dataset.csv')
X = dataset.iloc[:, 0:35].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 35)
X_pca = pca.fit(X)
explained_variance = pca.explained_variance_ratio_

#Scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-')
plt.plot([0, 36], [1-0.97, 1-0.97], 'k-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

# Visualise
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(50,100)
plt.grid()
plt.plot(var)