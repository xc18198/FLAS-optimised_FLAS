import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import datetime
import pdb
from mydata import *
from  model import *
import pylab as pl
np.random.seed(0)


USE_CUDA = True

def train_step(model,x,y,optimizer):
    model.train()
    # 梯度清零
    optimizer.zero_grad()
    # 正向传播求损失
    if USE_CUDA:
        x = x.cuda()
        y = y.cuda()

    # pdb.set_trace()
    pred = model(x)
    loss = F.mse_loss(pred,y)
    # # 反向传播求梯度
    loss.backward()
    optimizer.step()

    return loss.item()

def valid_step(model,x,y):
    model.eval()
    if USE_CUDA:
        x = x.cuda()
        y = y.cuda()
    pred = model(x)
    loss = F.mse_loss(pred,y)
    return loss.item()


def train(model,dl_train,dl_valid,epochs = 100,start=1,lr=1e-4):
    dfhistory = pd.DataFrame(columns = ["epoch","loss","val_loss"]) 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file = open('train.log','w')
    print("Start Training...")
    print("=========="*8 + "%s"%nowtime)
    log_step_freq = 10
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    total_step = 0
    for epoch in range(start,epochs+start):  
        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        for step, (x,y) in enumerate(dl_train, 1):
            # pdb.set_trace()
            total_step += 1
            loss = train_step(model,x,y,optimizer)

            # 打印batch级别日志
            loss_sum += loss
            # metric_sum += metric
            # pdb.set_trace()
            log_file.write('%d,%f\n'%(total_step,loss))
            if step%log_step_freq == 0:   
                print(("[step = %d] loss: %.3f") %
                      (step, loss_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        # val_metric_sum = 0.0
        val_step = 1

        for val_step, (x,y)  in enumerate(dl_valid, 1):
            val_loss= valid_step(model,x,y)
            val_loss_sum += val_loss
            # val_metric_sum += val_metric
            # print(("[val step = %d] loss: %.3f, mAP: %.3f") % (val_step, val_loss, val_metric))

        # pdb.set_trace()
        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum/step, 
                val_loss_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print("\nEPOCH = %d, loss = %.3f, val_loss = %.3f"%info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)
        if epoch % 10 == 0:
            torch.save(model.state_dict(),'weights/model_weights_%d_%0.3f.pth'%(epoch,val_loss_sum/val_step))
    return dfhistory


##
path = '10-Site_2.csv'
datalen = getdataInfo(path)
data_train = MyDataset(path,3,500000)
dl_train = DataLoader(data_train,batch_size = 64,shuffle = True,num_workers=0)

data_test = MyDataset(path,500000,91825)
dl_test = DataLoader(data_test,batch_size = 64,shuffle = False,num_workers=0)

model = Deep_model()

if USE_CUDA==True:
    model = model.cuda()

his = train(model,dl_train,dl_test,epochs = 20,start=1)



