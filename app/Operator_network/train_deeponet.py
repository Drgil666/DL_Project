import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
# 单自由度
import torch
import torch.nn as nn
from django.http import HttpRequest
from torch import optim

from app import views
from app.Operator_network.Models.deeponet import DeepOnet
from app.Operator_network.metrics import *

re = HttpRequest()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
dt = 0.001
min_t = 0
max_t = 5
a = np.arange(min_t,max_t,dt)
lt = a.shape[0]


def show_deeponet(df,seed,branch_layers,trunk_layers,activation,initializer,learning_rate,num_epochs):
    # 验证
    # print(seed,branch_layers,trunk_layers,activation,initializer,learning_rate,num_epochs)

    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time_train = time.time()

    model = DeepOnet(
        branch_layers,
        trunk_layers,
        activation,
        initializer
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),learning_rate)

    num_epochs = num_epochs

    '''load train dataset'''

    data_train = np.load(df,allow_pickle=True)

    branch_input = data_train['X'].astype(np.float32)
    branch_input = branch_input[:,0:lt:5]
    branch_input = torch.Tensor(branch_input).to(device)

    t = data_train['t'].astype(np.float32)
    len_t = len(t)
    trunk_input = t.reshape(len_t,1)
    trunk_input = trunk_input[:lt:5]
    trunk_input = torch.Tensor(trunk_input).to(device)

    labels_train = data_train['Y'].astype(np.float32)
    labels_train = torch.tensor(labels_train).to(device)
    labels_train = labels_train[:,0:lt:5]

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
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(branch_input,trunk_input).to(device)
        loss = criterion(outputs,labels_train).to(device)
        loss.backward()
        optimizer.step()

        if (epoch+1)%1000 == 0:
            loss_epoches.append(loss.item())
            # print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch + 1, num_epochs, loss.item()))
            callback.on_epoch_end(epoch+1,loss)
    views.loss_g = loss_epoches
    # end_time_train = time.time()
    # train_time = end_time_train-start_time_train

    # print('训练数据共花费时间为',train_time)
    print('loss数据',loss_epoches)

    # # # #save model
    model_path = os.path.join('app/model_temp/deeponet_zhouqi_5s_0.005'+str(int(datetime.now().timestamp()))+'.pth')
    torch.save(model.state_dict(),model_path)
    return model_path


def show_deeponet_test(data_path,model_path):
    '''load test dataset'''
    start_time = time.time()
    data_test = np.load(data_path,allow_pickle=True)
    t = data_test['t'].astype(np.float32)
    len_t = len(t)
    # cnt=6
    trunk_input_test = t.reshape(len_t,1)
    trunk_input_test = trunk_input_test[0:lt:5]
    trunk_input_test = torch.Tensor(trunk_input_test).to(device)

    branch_input_test = data_test['X'].astype(np.float32)
    branch_input_test = branch_input_test[:,0:lt:5]
    branch_input_test = torch.Tensor(branch_input_test).to(device)

    labels_test = data_test['Y'].astype(np.float32)
    labels_test = torch.tensor(labels_test).to(device)
    labels_test = labels_test[:,0:lt:5]

    test_nums = branch_input_test.shape[0]
    # print("test_nums是：",test_nums)

    model = DeepOnet(
        [1000,512,256,64],
        [1,128,128,64,64],
        "relu",
        "Glorot normal"
    ).to(device)
    model_path1 = model_path
    # model.load_state_dict(torch.load(model_path1))
    model.load_state_dict(torch.load(model_path1,map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        outputs_test = model(branch_input_test,trunk_input_test).to(device)

    end_time = time.time()
    test_time = end_time-start_time
    print('测试数据共花费时间为',test_time)

    labels_test_temp = labels_test.cpu().detach().numpy()
    outputs_test_temp = outputs_test.cpu().detach().numpy()

    l2_error = l2_relative_error(labels_test_temp,outputs_test_temp)
    mse = mean_squared_error(labels_test_temp,outputs_test_temp)
    R2 = R2_(labels_test_temp,outputs_test_temp)
    mae = MAE(labels_test_temp,outputs_test_temp)

    print('L2相对误差(l2_relative_error)为：',l2_error)
    print('均方误差(MSE)为',mse)
    print('R2为',R2)
    print('平均绝对误差为',mae)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 推荐使用SimHei字体显示中文
    plt.rcParams['axes.unicode_minus'] = False

    for i in range(0,test_nums):
        # plt.show()
        x_data = np.arange(0,max_t,0.005)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 推荐使用SimHei字体显示中文
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(9,12))

        plt.subplot(2,1,1)
        plt.plot(x_data,labels_test[i,:],label='系统真实响应值')
        plt.plot(x_data,outputs_test[i,:],label='响应值')
        # plt.plot(x_data,outputs_test1[i,:],  label='阶跃响应值')

        plt.xlabel('时间(s)')
        plt.ylabel('干扰力矩')
        plt.legend(fontsize='18')
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(x_data,branch_input_test[i,:],color='r',label='系统输入')
        plt.xlabel('时间(s)')
        plt.ylabel('角加速度')
        plt.legend(fontsize='18')
        plt.grid(True)

        # filename = f'H:\\DL\\app\\static\\img\\deep_png{i}.png'
        # 保存为图片
        plt.savefig(os.path.join('app/static/img/'+'deep_png'+str(i)+'.png'))
        plt.close()

    return l2_error,mse,R2,mae,test_nums


if __name__ == '__main__':
    show_deeponet()
