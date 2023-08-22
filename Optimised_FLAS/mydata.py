import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import pdb
import random
import pandas as pd
import torchvision as tv
import scipy.io as sio

class MNISTDataset(data.Dataset):
    def __init__(self, path,selectList=None,beta=2):
        self.data0=tv.datasets.MNIST('../data')
        self.beta = beta
        try:
            selectList.shape[0]>0
        except:
            selectList = range(self.data0.data.shape[0])

        self.data = self.data0.data[selectList].unsqueeze(1).float()/255.0
        self.target = self.data0.targets[selectList]
        self.selectList = selectList
        self.P = np.ones(len(selectList))
        self.join_num = np.ones(len(selectList))

    def __getitem__(self, index):
        return self.data[index],self.target[index]

    def __len__(self):
        return self.data.shape[0]

    def resetDatalist(self,selectList):
        self.data = self.data0.data[selectList].unsqueeze(1).float()/255.0
        self.target = self.data0.targets[selectList]

    def shufful(self):
        self.data = self.data0.data[self.selectList].unsqueeze(1).float()/255.0
        self.target = self.data0.targets[self.selectList]
        index = np.random.rand(len(self.selectList))<self.P
        self.join_num += index
        self.data = self.data[np.where(index==1)[0]]
        self.target = self.target[np.where(index==1)[0]]
        print('select %d data/ total %d data'%(len(self.data),len(self.selectList)))
    def setP(self,g_lambda,acc):
        # pdb.set_trace()
        self.P = self.beta*((acc+0.6)/g_lambda)/(np.log(0.6*self.join_num)+1)



        
class CIFAR10Dataset(data.Dataset):
    def __init__(self, path,selectList=None,beta=1.2):
        self.beta = beta

        self.data0 = []
        self.target0 = []
        self.transforms = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        for i in range(1,6):    
            data = sio.loadmat(os.path.join(path,'data_batch_%d.mat'%i))
            # pdb.set_trace()
            self.data0.append(data['data'].reshape([-1,3,32,32]).astype('float32')/255.0)
            self.target0.append(data['labels'].squeeze())

        data = sio.loadmat(os.path.join(path,'test_batch.mat'))
        self.data0.append(data['data'].reshape([-1,3,32,32]).astype('float32')/255.0)
        self.target0.append(data['labels'].squeeze())  
        self.data0 = np.concatenate(self.data0,axis=0)
        self.target0 = np.concatenate(self.target0, axis=0)

        # pdb.set_trace()
        try:
            selectList.shape[0]>0
        except:
            selectList = range(self.data0.shape[0])
        
        self.selectList = selectList
        self.data = torch.tensor(self.data0[selectList]).float()
        self.target = torch.tensor(self.target0[selectList])
        self.P = np.ones(len(selectList))
        self.join_num = np.ones(len(selectList))

    def __getitem__(self, index):
        return self.transforms(self.data[index]),self.target[index]

    def __len__(self):
        return self.data.shape[0]

    def resetDatalist(self,selectList):
        self.data = torch.tensor(self.data0[selectList]).float()
        self.target = torch.tensor(self.target0[selectList])

    # def setLambda(self,g_lambda,loss):
    #     index = np.where(loss<g_lambda)[0]
    #     # pdb.set_trace()
    #     self.data = self.data[index]
    #     self.target = self.target[index]

    def shufful(self):
        # pdb.set_trace()
        self.data = self.data0[self.selectList]
        self.target = self.target0[self.selectList]
        index = np.random.rand(len(self.selectList))<self.P
        self.join_num += index
        self.data = torch.tensor(self.data[np.where(index==1)[0]]).float()
        self.target = torch.tensor(self.target[np.where(index==1)[0]])

    def setP(self,g_lambda,loss):
        # pdb.set_trace()
        self.P = self.beta*(1.0-loss/g_lambda)/(np.log(self.join_num)+1)

def genSelectList(num_data,num_part,test_rate,IID=True,data=None):
    userdataList = []
    if IID==True:
        for i in range(num_part):
            index = np.random.rand(num_data)
            # pdb.set_trace()
            part_idx = np.where(index<1.0/num_part)[0]
            rand_idx = np.random.rand(len(part_idx))
            part_train_idx = np.where(rand_idx>test_rate)[0]
            part_test_idx = np.where(rand_idx<=test_rate)[0]
            userdataList.append((part_idx[part_train_idx],part_idx[part_test_idx]))
    elif IID==False:
        part_data_num = int(num_data/num_part)
        labels = data.target
        for i in range(num_part):
            cls_idx = np.random.permutation(10)[:7]
            index = torch.rand(num_data)
            for j in cls_idx:
                # pdb.set_trace()
                index += 0.2*(labels==j).float()

            # part_idx = np.where(index>p_th)[0]

            part_idx = torch.argsort(index,descending=True)[:part_data_num]

            # pdb.set_trace()
            rand_idx = np.random.rand(len(part_idx))
            part_train_idx = np.where(rand_idx>test_rate)[0]
            part_test_idx = np.where(rand_idx<=test_rate)[0]
            userdataList.append((part_idx[part_train_idx],part_idx[part_test_idx]))

    return userdataList


if __name__ == "__main__":
    data1 = CIFAR10Dataset('../data/CIFAR10')
    # data2 = MNISTDataset('../data')
    datalist1 = genSelectList(len(data1),5,0.2,IID=False,data=data1)
    

