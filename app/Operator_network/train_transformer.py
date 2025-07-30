from datetime import datetime

import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from torch import optim
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import argparse
from app.Operator_network.Models.transformer import TransformerClassifier
from app.Operator_network.metrics import *
from django.http import HttpRequest
from app import views

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def data_load(batch_size):
    # 数据预处理：定义图像的转换
    # 将图像尺寸变为1x28x28，并标准化至[-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,),(0.5,))  # 标准化到[-1, 1]之间
    ])
    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
    test_dataset = datasets.MNIST(root='./data',train=False,transform=transform,download=True)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,test_loader


train_loader,test_loader = data_load(batch_size=512)

#epoch=10
def show_transformer(df,seed,branch_layers,trunk_layers,activation,initializer,learning_rate,num_epochs,model_name):
    np.random.seed(seed)
    torch.manual_seed(seed)
    start_time_train = time.time()
    model = TransformerClassifier(embedding_dim=128,num_heads=16,num_layers=3,hidden_dim=256,num_classes=10,
                                  dropout_rate=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    loss_epoches = []

    class MyCallback:
        def __init__(self,num_epochs):
            self.start_time = None
            self.num_epochs = num_epochs  # Total number of epochs for estimate time


        def on_epoch_begin(self):
            self.start_time = time.time()
            views.epoch_g = 0
            views.logs_g = []

        def on_epoch_end(self,epoch,loss):
            views.epoch_g = epoch

            # Update logs globally
            all_log = {'epoch':views.epoch_g,'loss':loss.item()}
            views.logs_g.append(all_log)

            # Calculate estimated remaining time
            end_time = time.time()
            elapsed_time = end_time-self.start_time
            views.estimate_time = elapsed_time*(self.num_epochs-views.epoch_g)/(
                views.epoch_g)  # Estimate time remaining

            print(f'Epoch [{views.epoch_g}/{self.num_epochs}], Loss: {loss.item():.8f}')
            print(f'Estimated time remaining: {views.estimate_time/60:.2f} minutes')

    callback = MyCallback(num_epochs=num_epochs)
    callback.on_epoch_begin()  # Callback for the start of the epoch
    cnt = 0
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        last_loss = []
        for images,labels in train_loader:
            cnt = cnt+1
            images,labels = images.to(device),labels.to(device)

            # 前向传播
            outputs = model(images).to(device)
            loss = criterion(outputs,labels).to(device)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            last_loss = loss
            # if cnt % 100 ==0:
            # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        loss_epoches.append(running_loss/len(train_loader))
        callback.on_epoch_end(epoch+1,last_loss)
    views.epoch_g = views.epoch_g*1000
    views.loss_g = loss_epoches
    print('loss数据',loss_epoches)
    # # # #save model
    model_path = os.path.join('app/model_temp/'+'transformer_'+model_name+'_'+str(datetime.now().timestamp())+'.pth')
    torch.save(model.state_dict(),model_path)


def show_transformer_test(data_path,model_path):
    model = TransformerClassifier(embedding_dim=128,num_heads=16,num_layers=3,hidden_dim=256,num_classes=10,
                                  dropout_rate=0.1).to(device)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    # 测试模型
    model.eval()
    Accuracy = 0
    Precision = 0
    Recall = 0
    F1 = 0.0
    total = 0
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            Accuracy += (predicted == labels).sum().item()

    return float(100*Accuracy/total)
