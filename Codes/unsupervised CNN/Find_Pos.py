# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:38:51 2020

@author: Ngai Yueng
"""
import math
import io
from scipy.io import loadmat 
import numpy as np

from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as pyplot
from skimage.restoration import denoise_wavelet
from skimage.restoration import denoise_bilateral
from skimage.restoration import denoise_nl_means
from skimage.util import random_noise


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
from skimage import segmentation
import torch.nn.init
import torchvision.transforms as transforms





def Read_img(path):
    
    image = cv2.imread(path)
    background = image[10,10]
    
    return image, background
        


def find_pos(image, background):
    '''finding the signal by finding the center of mass'''
    
    counter = 0
    pos_x, pos_y = [0,0]
    for i in range(220):
        for j in range(220):
            R,G,B = image[i,j,:]
            if R == background[0] and G == background[1] and B == background[2]:
                pos_x += 0
                pos_y += 0
            else:
                counter += 1
                pos_x += i
                pos_y += j
    
    if counter == 0:
        counter = 1
    
    return pos_x/counter+146, pos_y/counter+146
        
    

def Read_raw(path,idx):
    
    a = loadmat(path)
    raw=(np.array(a['ImData'])[0:512*512,idx])
    raw = raw.reshape((512,512))
    sig = np.array(a['ImData'])[int(512*512+1), idx]
    x =   np.array(a['ImData'])[int(512*512+2), idx]
    y =   np.array(a['ImData'])[int(512*512+3), idx]
    return raw, sig, x, y
    
    
def FindSig(img,x_est,y_est):
    
    Window = img[int(round(x_est))-25:int(round(x_est))+25, int(round(y_est))-25:int(round(y_est))+25]
    indices = np.where(Window == Window.max())
    indices = np.array(indices)
    indices[0] += int(round(x_est))-25
    indices[1] += int(round(y_est))-25
    return indices, Window.max()
    
    


#%%

DataIdx = 200

Signal = np.zeros((DataIdx,2))
Orig = np.zeros((DataIdx,3))
NIS = 0 #number of identified signals 
correct = 0
for i in range(DataIdx):
    counter = 0
    posx, posy = 0, 0
    for j in range(3):
        
        path = 'Train/Tag'+ str(i+1)+'_out_600_2_'+str(j+1)+'.png'
        image, bg = Read_img(path)

        x, y = find_pos(image, bg)
        #print(x)
        #print(y)
        if x == 146 and y == 146:
            counter += 0
        else:
            counter += 1
            posx += x
            posy += y
        
    #print(counter)
    if counter == 0:
        Signal[i,0]=0
        Signal[i,1]=0
        
    else:
        Signal[i,0] = posx/counter
        Signal[i,1] = posy/counter
        
    print('estimated tag'+str(i+1)+' position: '+str(Signal[i,0]) + ' ,'+ str(Signal[i,1]))

    k = i//37
    l = i % 37
    rawpath = 'rawData/'+ str(k+1)+'.mat'    
    image, rsig, rx, ry = Read_raw(rawpath,l)
    if Signal[i,0] != 0:
        NIS += 1

        indices, sig = FindSig(image,Signal[i,0],Signal[i,1])
        Orig[i,0] = indices[0]
        Orig[i,1] = indices[1]
        Orig[i,2] = sig
        
        print('tag'+str(i+1)+' position: '+str(Orig[i,0]) + ', '+ str(Orig[i,1]))
        print('tag'+str(i+1)+' signal: '+str(Orig[i,2]))
        
        err = (Orig[i,0]-ry)**2 + (Orig[i,1]-rx)**2
        if err < 100:
            correct += 1
        
    print('Raw Data: '+ str(ry)+', '+str(rx)+ ', '+ str(rsig))
    print()
print(str(NIS)+' signals identified ' + str(correct) + ' correct')
tag = open('First200TagSigna2.txt','w')
for row in range(200):
    np.savetxt(tag, Orig[row,:])

tag.close()
