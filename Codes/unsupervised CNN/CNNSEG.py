# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:26:40 2020

@author: Yiyang Liu

Modified from https://github.com/kanezaki/pytorch-unsupervised-segmentation 
By Kanezaki
"""


import math
import time
from PIL import Image
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

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge


np.random.seed(42)
torch.random.seed()

#
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

#%%
#load data

def Prod(data):
    '''change the data to RGB image'''
    
    #image = loadmat(path)['tag'+i]
    #data = np.zeros(512*512*3)
    
    data = np.stack((data,)*3,-1)
    return data
    
    
    

def text_read(filename):
    '''read data from txt file'''
    
    
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
 
    rows=512
    datamat=np.zeros((512,512))
    row_count=0
    
    for i in range(rows):
        content[i] = content[i].strip().split(' ')
        datamat[row_count,:] = content[i][:]
        row_count+=1
 
    file.close()
    return datamat


#%%

class CNNSeg(nn.Module):
    '''CNN Segmentation '''
    
    def __init__(self, dim):
        '''The structure: Convolution - Batch Normalization - ... Convolution - Batch Normalization '''
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim, config['number_channel'], kernel_size=3, stride=1, padding=1)  # convolutional layer 1
        self.normal1 = nn.BatchNorm2d(config['number_channel'])
        self.conv = nn.ModuleList()
        self.normal = nn.ModuleList()
        for i in range(config['number_conv']-1): # convolution layer 2 to n
            self.conv.append(nn.Conv2d(config['number_channel'], config['number_channel'], kernel_size=3, stride=1, padding=1))
            self.normal.append(nn.BatchNorm2d(config['number_channel']))
                
        self.convn = nn.Conv2d(config['number_channel'], config['number_channel'], kernel_size=3, stride=1, padding=1)  # convolutional layer 3 n+1
        self.normaln = nn.BatchNorm2d(config['number_channel'])
        #self.init_weights()  We don't initialize weights because we are running on on image at a time.

    
    def init_weights(self):

        C_in = self.conv1.weight.size(1)
        nn.init.normal_(self.conv1.weight, 0.0, 1 / math.sqrt(5 * 5 * C_in))
        nn.init.constant_(self.conv1.bias.double(), 0.0)
        for conv in self.conv:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / math.sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)
        C_in = self.convn.weight.size(1)
        nn.init.normal_(self.convn.weight, 0.0, 1 / math.sqrt(5 * 5 * C_in))
        nn.init.constant_(self.convn.bias, 0.0)

            

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.normal1(x)
        for i in range(config['number_conv']-1):
            x = F.relu(self.conv[i](x))
            x = self.normal[i](x)
        x = F.relu(self.convn(x))
        x = self.normaln(x)
        return x



def train(config, dataset, model, idx, tag):
    
    
    global T


    if 'use_weighted' not in config:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(torch.tensor([1.0,20.0]))
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    #set label colors.
    if config['colorful_labels']:
        label_colors = np.array(((50,25,46),(255,187,175)))
    else:
        label_colors = np.random.randint(255,size=(config['number_channel'],3))
    
    
    
    #The following code from kanezaki
    for batch_idx in range(config['max_iter']):
    # forwarding
        optimizer.zero_grad()
        output = model(data)[0]
        #print(batch_idx)
        output = output.permute(1,2,0).contiguous().view( -1, config['number_channel'] )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))


    
    
    im_target_rgb = np.array([label_colors[ c % 100 ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape((300,300,3)).astype( np.uint8 )
    
    
    #some noises at the edge
    for i in range(40):
        im_target_rgb =np.delete(im_target_rgb, (300-2*i-1), axis =0)
        im_target_rgb =np.delete(im_target_rgb, (0), axis =0)
        im_target_rgb =np.delete(im_target_rgb, (300-2*i-1), axis =1)
        im_target_rgb =np.delete(im_target_rgb, (0), axis =1)
        
    
    
    '''
    print('Finished Training \n')
    im1 = [[50,25,46] for _ in range(296)]
    im1 = [im1 for _ in range(296)]
    im1 = np.array(im1)
    im2 = [[255,187,175] for _ in range(296)]
    im2 = [im2 for _ in range(296)]
    im2 = np.array(im1)
    
    if (im_target_rgb - im1).any() != 0:
        if (im_target_rgb - im2).any != 0:
            T = True
            '''
            
    filename = 'Train/Tag'+tag+'_out_600_2_'+str(idx +1)+'.png'
    cv2.imwrite(filename,im_target_rgb)
    cv2.waitKey(10)
    

    




if __name__ == '__main__':

    
    
    
    ini= 50
    set = 150
    start = time.time()
    for j in range(ini,ini+set):
    
    
        tag_number = str(j+1)
    
    # define config parameters for training
        config = {
            'dataset_path': 'rawData/1.mat',
            'txt_path': 'processedData/tag'+tag_number+'.txt',
            'number_conv':2,
            'learning_rate': 0.1,           # learning rate
            'max_iter': 150,
            'number_channel': 2,
            'colorful_labels': 1,  #if we want the labels to be rbg, change the value to 1
            'save': 0 #whether to save the image if yes change to 1
        
        }
    # create dataset
        model = CNNSeg(1)
        global T
        T = False
        for i in range(3):
                
            
            print('tag_'+str(j+1))
            '''
            dataset = LoadData(config['dataset_path'],i)
            data = PreProcess(dataset)'''
            data = text_read(config['txt_path'])
            data = Prod(data)
            
            #print(data[9,5,:])
            

            datain = data*255
            #data = data*3
            if i ==1:
                filename = 'Train/Tag'+tag_number+'_in_200_2_'+str(i+1)+'.png'
                cv2.imwrite(filename,datain)
            if data.max() < 1:
                data = data*5
            data = Image.fromarray(data.astype('uint8'), 'RGB')
            transfrom = transforms.Compose([transforms.CenterCrop(300), transforms.ToTensor()])
            data = transfrom(data)
        
  
            data = data.unsqueeze(0).float()

            
            data = Variable(data)
        #data = data.cuda()

            model = CNNSeg(data.size(1))
            #model = model.cuda()   #for gpu
            # train our model on dataset
            train(config, data.float(), model, i, tag_number)
    print(time.time()-start)
