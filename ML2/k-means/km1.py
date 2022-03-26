#https://deepblade.com/artificial-intelligence/machine-learning/k-means-clustering-algorithm/

# load dataset
import pandas as pd
data = pd.read_csv('Mall_Customers.csv')
#print(data.sample(5))
#input()

data = data[['Annual Income (k$)','Spending Score (1-100)']]

# rename names of columns for simplicity
data = data.rename(columns={'Annual Income (k$)': 'income', 'Spending Score (1-100)': 'score'})

print(data.sample(5))
input()

# visualize the data distribution
import matplotlib.pyplot as plt
plt.scatter(data['income'],data['score'])
plt.show()
#input()

#Choose a suitable number of clusters (K)
# calculate sum of squares errors for different K values

from sklearn.cluster import KMeans
print("KMeans")
k_values = [1,2,3,4,5,6,7,8,9,10]
wcss_error = []
for k in k_values:
   #print(k)
   model = KMeans(n_clusters=k)
   model.fit(data[['income','score']])
   wcss_error.append(model.inertia_)

input("WCSS")

# sum of squares error for K=1 to k=10
print(wcss_error)

# plot WCSS error corresponding to different K values
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS Error')
plt.plot(k_values, wcss_error)

# train model using k=5
model = KMeans(n_clusters=5)
pred = model.fit_predict(data[['income','score']])
print(pred)

# add cluster column to dataset
data['cluster'] = pred
data.sample(5)


# centers of clusters
model.cluster_centers_

'''
OUTPUT:
array([[55.2962963 , 49.51851852],
       [25.72727273, 79.36363636],
       [86.53846154, 82.12820513],
       [26.30434783, 20.91304348],
       [88.2       , 17.11428571]])
'''
# visualize clusted data

cluster1 = data[data['cluster']==0]
plt.scatter(cluster1['income'], cluster1['score'])

cluster2 = data[data['cluster']==1]
plt.scatter(cluster2['income'], cluster2['score'])

cluster3 = data[data['cluster']==2]
plt.scatter(cluster3['income'], cluster3['score'])

cluster2 = data[data['cluster']==3]
plt.scatter(cluster2['income'], cluster2['score'])

cluster3 = data[data['cluster']==4]
plt.scatter(cluster3['income'], cluster3['score'])

plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color='black')


