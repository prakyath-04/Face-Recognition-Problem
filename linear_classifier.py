import os,glob,PIL,sys,math
import numpy
from PIL import Image
from numpy import linalg as la

train_file = sys.argv[1]
test_file = sys.argv[2] 
im,img,classes,test_im, im_t = [],[],[],[],[]

with open(train_file) as fl:
    for i in fl:
        temp = i[:-1] # removing new line
        t1 = i.split() 
        img.append(t1[0]) # the first 21 characters of a line in the file corresponds to the image path 
        classes.append(t1[1]) # the class of the image 

with open(test_file) as fl:
    for i in fl:
        temp = i[:-1]
        test_im.append(temp)
#image object to mat
rsize = 64
l1 = len(img)
for i in range(l1):
    n= Image.open(img[i]) 
    n = n.convert('L') 
    sz = n.size
    ratio = float(sz[1] / sz[0])
    new_ht = rsize
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
x_mat = x_mat - mn 
x_mat_t = x_mat.T
sigma = (numpy.matmul(x_mat_t,x_mat))  # the covariance matrix
u,s,v = la.svd(x_mat)
eig_vec = v.T
# x_mat = x_mat +mn
train_vec_1 = numpy.matmul(x_mat,eig_vec)
train_vec_1 = train_vec_1[:,:32]
t = numpy.ones((train_vec_1.shape[0],train_vec_1.shape[1] +1))
t[:,:train_vec_1.shape[1]] = train_vec_1
train_vec = t
if la.norm(train_vec) != 0:
    train_vec = train_vec / abs(la.norm(train_vec))
#######################################################################
l1 = len(test_im)
for i in range(l1):
    n= Image.open(test_im[i]) 
    n = n.convert('L') 
    sz = n.size
    ratio = float(sz[1] / sz[0])
    new_ht = rsize
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
m1 = numpy.mean(test_mat,axis=0)
test_mat = test_mat - mn
test_class = []
test_vec_1 = numpy.matmul(test_mat,eig_vec[:,:32])
t = numpy.ones((test_vec_1.shape[0],test_vec_1.shape[1] +1))
t[:,:test_vec_1.shape[1]] = test_vec_1
test_vec = t
######################################################################
class_dup = classes  
classes= [] 
for i in class_dup: 
    if i not in classes: 
        classes.append(i)
l1 = len(class_dup)
l2 = len(classes)
grp = {}
l1 = len(class_dup)
l2 = len(classes)
for i in classes:
    grp[i] = [] 
for i in range(l1):
    grp[(class_dup[i])].append((train_vec[i,:]))
######################################################################
n = 1
w = numpy.zeros((len(classes),test_vec.shape[1]))
for j in range(len(classes)):
    for i in range(12000):
        t=0 
        for k in range(numpy.asarray(grp[classes[j]]).shape[0]):
            dnr = float(0)
            mx =-10**9
            for l in range(w.shape[0]):
                if mx < numpy.matmul(w[l],train_vec[k]):
                    mx =  numpy.matmul(w[l],(grp[classes[j]][k]).T )
                    mx_ind = l
            for l in range(w.shape[0]):
                d = float(math.exp((float(numpy.matmul(w[l], (grp[classes[j]][k]).T )))))
                dnr = float(dnr + d)
            nmr = float(math.exp(numpy.matmul(w[j],(grp[classes[j]][k]).T)))
            p = float(nmr / dnr)
            t = t + (1-p) * (grp[classes[j]][k])
        w[j] = w[j] + n*t 
##############################################
prob = numpy.matmul(w,test_vec.T)
# print(prob)
for i in range(test_vec.shape[0]):
    mx = -10**9
    for j in range(len(classes)):
        if mx < prob[j,i]:
            mx = prob[j,i]
            mx_ind = j
    print(classes[mx_ind])


