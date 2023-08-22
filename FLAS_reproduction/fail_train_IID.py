import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import datetime
import pdb
from mydata import *
from model_server import *

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
    loss = F.cross_entropy(pred,y.long())
    acc = torch.mean((pred.argmax(dim=1)==y.long()).float())
    # # 反向传播求梯度
    loss.backward()
    optimizer.step()

    return loss.item(),acc.item()

def valid_step(model,x,y):
    model.eval()
    if USE_CUDA:
        x = x.cuda()
        y = y.cuda()
    pred = model(x)
    loss = F.cross_entropy(pred,y.long())
    acc = torch.mean((pred.argmax(dim=1)==y.long()).float())
    return loss.item(),acc.item()

def check_data_loss(model,dl_train):
    model.eval()
    total = []
    for x,y in dl_train:
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()
        pred = model(x)
        loss = F.cross_entropy(pred,y.long()).cpu().detach().item()
        total.append(loss)
    # pdb.set_trace()
    total = np.array(total)
    return total

def train(model_name,model,data_train,data_valid,global_epoch,local_epochs = 100,lr=1e-4,batch_size=64):
    dl_train = DataLoader(data_train,batch_size = batch_size,shuffle = True,num_workers=0)
    dl_valid = DataLoader(data_valid,batch_size = batch_size,shuffle = True,num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(local_epochs):  
        # 1，训练循环-------------------------------------------------
        # train_loss = []
        for step, (x,y) in enumerate(dl_train, 1):
            loss,acc = train_step(model,x,y,optimizer)
            train_loss.append(loss)
            train_acc.append(acc)

        # 2，验证循环-------------------------------------------------
        # val_loss = []
        for val_step, (x,y)  in enumerate(dl_valid, 1):
            loss,acc = valid_step(model,x,y)
            val_loss.append(loss)
            val_acc.append(acc)

        # 打印epoch级别日志
        print("\t[model %s][local epoch %d]:train loss = %f, acc = %f, val loss = %f val acc = %f"%(model_name,epoch,np.mean(train_loss),np.mean(train_acc),np.mean(val_loss),np.mean(val_acc),))
    total_loss = (np.mean(train_loss),np.mean(val_loss))
    return total_loss


##

models = []
n_models = 3
model_server = Model_server()
for i in range(n_models):
    model = client_model_lenet5_large()
    if USE_CUDA==True:
        model = model.cuda()
    models.append(['model_%d'%i,model])
    model_server.append_client('model_%d'%i)

data = CIFAR10Dataset('../data/CIFAR10')
datalist = genSelectList(len(data),n_models,0.2,IID=True,data=data)

traindata = {}
testdata = {}
for i in range(len(datalist)):
    traindata[models[i][0]] = CIFAR10Dataset('../data/CIFAR10',datalist[i][0])
    testdata[models[i][0]] = CIFAR10Dataset('../data/CIFAR10',datalist[i][1])


## pretraining each other
local_train_max_epoch = 2

for l_epoch in range(local_train_max_epoch):
    print("local traing:")
    for model_no,(model_name,model) in enumerate(models):
        loss = train(model_name,model,traindata[model_name],testdata[model_name],l_epoch,local_epochs = 1,lr=1e-4)

        model_server.upload_local_weight(model_name,model.state_dict(),loss,False)
        print("[l_epoch %d]: local model %s training, train_loss = %0.2f,val_loss = %0.2f"%(l_epoch,model_name,loss[0],loss[1]))

model_server.update_global_model()
## FAIL Training
g_max_epoch = 100
for g_epoch in range(g_max_epoch):
    local_models = {}
    global_loss = {}
    for model_no,(model_name,model) in enumerate(models):
        #1 get global_model weight
        local_models[model_name] = model
        print("[g_epoch %d]: local model %s download global model weights"%(g_epoch,model_name))
        local_models[model_name].load_state_dict(model_server.get_Global_weight())

        #2 adaptive data sampling
        print("[g_epoch %d]: cal adaptive data sampling lambda"%g_epoch)
        traindata[model_name].resetDatalist(datalist[model_no][0])
        # pdb.set_trace()
        dl_train = DataLoader(traindata[model_name],batch_size = 1,shuffle = False,num_workers=0)
        global_loss[model_name] = check_data_loss(local_models[model_name],dl_train)
        model_server.upload_global_loss(model_name,global_loss[model_name].mean())
        print("[g_epoch %d]: local model %s upload local lambda"%(g_epoch,model_name))

    global_lambda = model_server.get_lambda()

    for model_no,(model_name,model) in enumerate(models):
        traindata[model_name].setLambda(global_lambda,global_loss[model_name])
        #3 local training
        print("[g_epoch %d]: model(%s) local training"%(g_epoch,model_name))
        loss = train(model_name,model,traindata[model_name],testdata[model_name],g_epoch,local_epochs = 20,lr=1e-4)
        print("[g_epoch %d]: model(%s) local training finish, loss = %f"%(g_epoch,model_name,loss[-1]))
        model_server.upload_local_weight(model_name,model.state_dict(),loss)
        # print("[g_epoch %d]: model(%s) upload local weights"%(g_epoch,model_name))

    #4 Adaptive Client Sampling and update global weights
    print("[g_epoch %d]:model server update global model weights"%g_epoch)
    model_server.update_global_model()



