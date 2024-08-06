# -*- coding: utf-8 -*-
"""
@author: Lingxiao Zhou
"""
from scipy.io import loadmat 
import numpy as np
import time
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as pyplot


from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from skimage.restoration import denoise_nl_means
from skimage import morphology
from skimage import segmentation
from sklearn.cluster import KMeans

import os
os.chdir('C:/Users/Lingxiao Zhou/Desktop/EECS 545 ML/final project/Data/using data')

# data split
def split_data(data_list, y_list, ratio=0.30):
    '''
    split data by given ratio
    '''
    X_train, X_test, y_train, y_test = train_test_split(data_list, y_list, test_size=ratio, random_state=42)
    print ('--------------------------------data shape-----------------------------------')
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    return X_train, X_test, y_train, y_test

def find_position(img):
    position=np.array([0,0])
    k=0
    mv=np.max(img)
    mi=np.min(img)
    for i in range(0, np.shape(img)[0]):
        for j in range(0,np.shape(img)[1]):
            if (img[i,j]>0.95*mv and img[i,j]>mi):
                position=position+[i,j]
                k=k+1
    if(k>1):
        position=position/k
    else:
        position=np.array([0,0])
        print("no bright light spot")
    return position.astype(int)

#%%
#loaddata: unsupervise withou data split
m=512*512
allimage=np.array(np.zeros((1,m)))
norm_i=np.zeros(1)

for i in range(1,41):
    alldata = loadmat(str(i)+'.mat')
    image=(np.array(alldata['ImData'])[0:m,:]).T
    feature= (np.array(alldata['ImData'])[m:m+4,:]).T
    norm_I=feature[:,1]
    
    allimage=np.vstack((allimage, image))
    norm_i=np.append(norm_i, norm_I)
    
allimage=np.delete(allimage, 0, axis=0)
norm_i=np.delete(norm_i, 0)

#%%

##################################################################
                     # preprocessing 0-1
##################################################################


scaler=preprocessing.MinMaxScaler().fit(allimage)
allimage=scaler.transform(allimage)


#%%
##################################################################
         # use k-means and denoising to find center
##################################################################
start = time.time()
#change the picture sample # here:
n=46

sample = np.reshape(allimage[n,:],(512,512))
print('original picture')
pyplot.imshow(sample)
pyplot.show()
###################################
preimage=[]
for i in range(0,m):
    preimage.append([allimage[n,i],allimage[n,i],allimage[n,i]])
preimage=np.mat(preimage)


iters=5
result=np.zeros((512,512))
kernel1 = morphology.disk(1)
kernel2 = morphology.disk(5)
kernel3 = morphology.disk(6)
for i in range(0,iters): 
    km=KMeans(n_clusters=2)
    km.fit(preimage)
    label =km.fit_predict(preimage)
    label=label.reshape([512,512])
    
    #denoise by open
    img_erosion =morphology.erosion(label, kernel1)
    img_dialtion = morphology.dilation(img_erosion, kernel2)
    
    # denoise by non-local mean filter
    denoise = denoise_nl_means(img_dialtion, 10, 10, 0.1)
    
    result=result+denoise
result=result/iters
result=morphology.erosion(result, kernel3)
result=morphology.dilation(result, kernel3)
print('center picture')
pyplot.imshow(result)
pyplot.show()

print('estimated certer position',find_position(result))

end = time.time()
print ('time:', end-start)
