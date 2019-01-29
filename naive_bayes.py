import os,glob,PIL,sys,math
import numpy
from PIL import Image
from numpy import linalg as la

train_file = sys.argv[1]
test_file = sys.argv[2] 
im,img,classes,test_im, im_t = [],[],[],[],[]

with open(train_file,'rt') as fl:
    
    for i in fl:
        temp = i[:-1] # removing new line
        t1 = i.split() 
        img.append(t1[0]) # the first 21 characters of a line in the file corresponds to the image path 
        classes.append(t1[1]) # 

with open(test_file,'rt') as fl:
    for i in fl:
        temp = i[:-1]
        test_im.append(temp)
#image object to mat
imsiz = 64  
l11 = len(img)
for i in range(l11):
    n= Image.open(img[i]) 
    n = n.convert('L') 
    sz = n.size
    ratio = float(sz[1] / sz[0])
    new_ht = imsiz
    new_wdt = int(math.floor(new_ht * ratio)) 
    n = n.resize((new_ht,new_wdt),Image.ANTIALIAS)
    t = numpy.array(n) 
    im.append(t)

num = len(im)
siz = im[0].shape
n_dim = siz[0] * siz[1]
x_mat = numpy.zeros((num,n_dim))
for i in range(num):
    x_mat[i] = im[i].flatten() 
###################################
#pca
mn = numpy.mean(x_mat,axis=0)
# print(mn,x_mat.shape)
x_mat = x_mat - mn 
x_mat_t = x_mat.T
sigma = (numpy.matmul(x_mat_t,x_mat)) 
# eig_val,eig_vec = la.eig(sigma)
# index = eig_val.argsort()[::-1]
# eig_val = eig_val[index]
# eig_vec = eig_vec[:,index]
u,s,v = la.svd(x_mat)
eig_vec = v.T
# print(eig_vec.shape)
x_mat = x_mat 
train_vec = numpy.matmul(x_mat,eig_vec)
train_vec = train_vec[:,:32]
# print(train_vec.shape)
####################################
class_dup = classes  
classes= [] 
for i in class_dup: 
    if i not in classes: 
        classes.append(i)
grp = {}
mn_var = {}
l1 = len(class_dup)
l2 = len(classes)
l3 = train_vec.shape[1]
# if l2 > l1:
#     l2 =l1
for i in classes:
    grp[i] = [] 
for i in range(l1):
    grp[(class_dup[i])].append((train_vec[i,:]))
for i in range(l2):
    mn_var[i,0] = []
    mn_var[i,1] = []
for i in range(l2):
    mn_var[i,0] = numpy.mean(numpy.array(grp[classes[i]]),axis= 0) # column wise mean
    mn_var[i,1] = numpy.var(numpy.array(grp[classes[i]]),axis= 0)
#######################################################################
l1 = len(test_im)
for i in range(l1):
    n= Image.open(test_im[i])
    n = n.convert('L') 
    sz = n.size
    ratio = float(sz[1] / sz[0])
    new_ht = imsiz
    new_wdt = int(math.floor(new_ht * ratio)) 
    n = n.resize((new_ht,new_wdt),Image.ANTIALIAS)
    t = numpy.array(n)
    im_t.append(t)

num = len(im_t)
siz = im_t[0].shape
n_dim = siz[0] * siz[1]
test_mat = numpy.zeros((num,n_dim))
for i in range(num):
    test_mat[i] = im_t[i].flatten() 
# print(test_mat)
m1 = numpy.mean(test_mat,axis=0)
test_mat = test_mat - mn
#######################################################3
test_class = []
test_ft = numpy.matmul(test_mat,eig_vec[:,:32])
l1 = test_ft.shape[0]
l2 = test_ft.shape[1]
l3 = len(classes)
ind_prob = numpy.zeros((1,l2))
tot_prob = numpy.zeros((1,l3))
for i in range(l1):
    mx = -10**9
    for j in range(l3):
        tot_prob[0,j] = 1
        for k in range(l2):
            ind_prob[0,k] = math.exp((-(test_ft[i][k] - mn_var[j,0][k])**2)/(2*(mn_var[j,1][k]))) / ((2*3.1415*(mn_var[j,1][k]))**.5)
            tot_prob[0,j] = tot_prob[0,j] * ind_prob[0,k]
    for j in range(l3):   
        if mx < tot_prob[0,j]:
            mx = numpy.max(tot_prob[0,j])
            mx_ind = j
    # mx_ind = tot_prob.index(mx)
    test_class.append(classes[mx_ind])

for i in range(len(test_class)):
    print(test_class[i])    


