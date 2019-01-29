import matplotlib 
import matplotlib.pyplot as plt 
import math 
import numpy as np 
import os
import glob
import PIL	
from PIL import Image
from numpy import linalg as la
from mpl_toolkits.mplot3d import Axes3D

im = []
for i in glob.glob("./dataset/*.jpg"):
    n= Image.open(i) # reading images from the dataset folder
    n = n.convert('L') 
    sz = n.size
    ratio = float(sz[1] / sz[0])
    new_ht = 64
    new_wdt = int(math.floor(new_ht * ratio)) 
    n = n.resize((new_ht,new_wdt),Image.ANTIALIAS)
    t = np.array(n) # resizing the image input and storing it in an array
    im.append(t)

num = len(im)
siz = im[0].shape
n_dim = siz[0] * siz[1]
x_mat = np.zeros((num,siz[0]*siz[1]))

for i in range(num):
	x_mat[i] = im[i].flatten() 

mn = np.mean(x_mat,axis=0)
x_mat = x_mat - mn 
x_mat_t = np.transpose(x_mat)

sigma = (np.matmul(x_mat_t,x_mat)) # the covariance matrix

u,s,v = la.svd(sigma,full_matrices=True)
eig_vec = np.transpose(v)
x_mat = x_mat +mn
new_ft_vec = np.matmul(x_mat,eig_vec) 

error = []		 										
n = list(range(1,250,5)) # 1000 eigen vectors seperated by 5 eigen vectors
for j in n:																##j eigen vectors 
	data_recon = np.matmul(new_ft_vec[:,:j],np.transpose(eig_vec[:,:j]))
	err = 0
	for i in range(num):								
		ans = data_recon[i,:] - x_mat[i,:]
		ans = np.square(la.norm(ans)) / np.square(la.norm(x_mat[i,:])) 
		err = err + ans
	error.append(err/num)

plt.plot(n,error) #Plotting the graph with MSE vs number of components
plt.xlabel ('Number of Principal components used')
plt.ylabel ('MSE of the images')
plt.title('MSE vs number of Principal components')
plt.show()

plt.scatter(new_ft_vec[:,0].tolist(),np.zeros(len(new_ft_vec[:,1])))
plt.xlabel('first PCA component')
plt.title('Image cluster in 1d')
plt.show()
plt.scatter(new_ft_vec[:,0].tolist(),new_ft_vec[:,1].tolist(),c=['r','b'])
plt.xlabel('First PCA component')
plt.ylabel('Second PCA component')
plt.title('Image cluster in 2d')
plt.show()
fig = plt.figure()
ax = Axes3D(fig)
x,y,z= (new_ft_vec[:,0].tolist(), new_ft_vec[:,1].tolist(), new_ft_vec[:,2].tolist())
color = ("red", "green","blue")
ax.scatter(x, y, z,c=color)
plt.title('Image cluster in 3d')
ax.set_xlabel('First PCA component')
ax.set_ylabel('Second PCA component')
ax.set_zlabel('Third PCA component')
plt.show()
