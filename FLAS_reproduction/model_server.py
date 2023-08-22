import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from sklearn import preprocessing
import torchvision as tv
import pdb

def weight_op(op,a,b=None):
    out = type(a)()
    if op == 'a+b':
        keys = a.keys()
        for key in keys:
            out[key] = a[key]+b[key]
    elif op == 'a*C':
        keys = a.keys()
        for key in keys:
            out[key] = a[key]*C
    elif op == 'mean':
        first = True
        out = type(a[0])()
        # pdb.set_trace()
        for model in a:
            keys = model.keys()
            if first:
                for key in keys:
                    out[key] = model[key]
                first = False
            else:
                for key in keys:
                    out[key] += model[key]

        n = len(a)
        for key in keys:
            out[key] /= n

    return out



class Model_server():
    def __init__(self,max_epoch = 20, step = 100,alpha_selectModel=0.4):
        self.clients = []
        self.g_models = None
        self.client_weights = {}
        self.clients_train_loss = {}
        self.clients_test_loss = {}
        self.global_loss = {}
        self.global_acc = {}
        self.cur_train_epoch = 0
        self.train_step_perEpoch = step
        self.max_epoch = max_epoch
        self.alpha_selectModel = alpha_selectModel

    def append_client(self,model_name):
        self.clients.append(model_name)
        self.client_weights[model_name] = 0
        self.clients_train_loss[model_name] = []
        self.clients_test_loss[model_name] = []
        self.global_loss[model_name] = []
        self.global_acc[model_name] = []


    def get_TrainInfo(self):
        return self.max_epoch,self.train_step_perEpoch

    def get_Global_weight(self):
        return self.g_models

    def upload_local_weight(self,model_name,weights,loss,flag=True):
        # pdb.set_trace()
        self.clients_train_loss[model_name].append(loss[0])
        self.clients_test_loss[model_name].append(loss[1])
        if flag == True:
            # 判别是否满足模型筛选的条件，决定是否需要上传权重
            L = np.max(self.clients_train_loss[model_name])
            dL = self.clients_train_loss[model_name][-1]-self.clients_train_loss[model_name][-2]
            if self.clients_train_loss[model_name][-1]<L*np.exp(self.alpha_selectModel*dL):
                self.client_weights[model_name]=weights
                print("model %s upload weights"%model_name)
        else:
            self.client_weights[model_name]=weights
            print("model %s upload weights"%model_name)

    def upload_global_loss(self,model_name,loss,acc):
        self.global_loss[model_name].append(loss)
        self.global_acc[model_name].append(acc)

    def get_lambda(self):
        sum_loss = 0
        for model_name in self.clients:
            sum_loss += self.global_loss[model_name][-1]
        return sum_loss/len(self.clients)*1.2

    def update_global_model(self):
        g_models = 0
        num_models = 0
        combile_models = []
        for model_name in self.clients:
            L = np.max(self.clients_train_loss[model_name])
            dL = self.clients_train_loss[model_name][-1]-self.clients_train_loss[model_name][0]
            # pdb.set_trace()
            if self.clients_train_loss[model_name][-1]<L*np.exp(self.alpha_selectModel*dL):
                combile_models.append(self.client_weights[model_name])
        # pdb.set_trace()
        self.g_models = weight_op('mean',combile_models)



class client_model_lenet5(nn.Module):
    def __init__(self):
        super(client_model_lenet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # pdb.set_trace()
        self.out = nn.Linear(256,10)

    def forward(self,x):
        # pdb.set_trace()
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        # x = F.relu(self.conv4(x))
        # x = self.pool4(x)
        # pdb.set_trace()
        out = F.softmax(self.out(x.reshape([-1,256])),dim=1)

        return out


class client_model_lenet5_large(nn.Module):
    def __init__(self):
        super(client_model_lenet5_large, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out = nn.Linear(256,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2b(F.relu(self.conv2(x))))
        x = self.pool2(x)
        x = F.relu(self.conv3b(F.relu(self.conv3(x))))
        x = self.pool3(x)
        # pdb.set_trace()
        x = F.relu(self.conv4(x))
        # x = self.pool4(x)
        # pdb.set_trace()
        out = F.softmax((self.out(x.reshape([-1,256]))),dim=1)

        return out

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.vgg11_feature = tv.models.vgg11().features[:16]
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.vgg11_feature.load_state_dict(torch.load('vgg_feature_base.pth'),False)
        # self.linear = nn.Linear(2048,512)
        self.out = nn.Linear(512,10)

    def forward(self,x):
        x = self.vgg11_feature(x)
        # pdb.set_trace()
        # x = self.linear(x.view(-1,2048))
        x = self.pool(x)
        x = self.out(x.view(-1,512))
        out = F.softmax(x,dim=1)

        return out


if __name__ == '__main__':
    model1 = client_model_lenet5()
    model2 = client_model_lenet5_large()
    model3 = vgg()
    x = torch.rand(16,3,32,32)
    py = model3(x)
    