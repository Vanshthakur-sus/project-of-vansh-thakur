import numpy as np
import pandas as pd

np.random.seed(23)

mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)

df = pd.DataFrame(class1_sample, columns=['feature1','feature2','feature3'])
df['target'] = 1

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)

df1 = pd.DataFrame(class2_sample, columns=['feature1','feature2','feature3'])
df1['target'] = 0

# Correct way to combine DataFrames
df = pd.concat([df, df1], ignore_index=True)

# Shuffle and reset the index
df = df.sample(40).reset_index(drop=True)

df.head()
import plotly.express as px
fig = px.scatter_3d(df, x=df['feature1'], y=df['feature2'],z=df['feature3'], color=df['target'].astype('str'))
fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarklateGrey')), selector=dict(mode='makers'))

fig.show()
# Apply Standard Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df.iloc[:,0:3]= scaler.fit_transform(df.iloc[:,0:3])
# Step2 - Find the Covaraince Matrix
covariance_matrix = np.cov([df.iloc[:,0],df.iloc[:,1],df.iloc[:,2]])
print('Covariance Matrix:\n', covariance_matrix)

# Step - Find EV and EV
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
eigen_values
eigen_vectors
%pylab inline


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
         FancyArrowPatch.__init__(self,(0,0),(0,0), *args, **kwargs)
         self._verts3d = xs,ys,zs


    def draw(Self, renderer):
      xs3d,ys3d,zs3d = self._verts3d
      xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
      self.set_positions((xs[0],ys[0],(xs[1], ys[1])))
      FancyArrowPatch.draw(self, renderer)


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')


ax.plot(df['feature1'],df['feature2'],df['feature3'],'o', markersize=8, color='blue',alpha=0.2)
ax.plot([df['feature1'].mean()],[df['feature2'].mean()], [df['feature3'].mean()],'o', markersize=10, color='red',alpha=0.5)
for v in eigen_vectors.T:
  a = Arrow3D([df['feature1'].mean(), v[0]],[df['feature2'].mean(), v[1]], [df['feature3'].mean(), v[3]])
  ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')


plt.title('Eigenvectors')


plt.show()
