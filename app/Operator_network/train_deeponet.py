# 单自由度
import os
import time

import torch
import torch.nn as nn
from torch import optim

from app.Operator_network.Models.deeponet import DeepOnet
from app.Operator_network.metrics import *


# re=HttpRequest()

def show_deeponet():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    seed = 12
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time_train = time.time()

    model = DeepOnet(
        [1000,512,256,64],
        [1,128,128,64,64],
        "relu",
        "Glorot normal"
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    num_epochs = 27000

    dt = 0.001
    min_t = 0
    max_t = 5
    a = np.arange(min_t,max_t,dt)
    lt = a.shape[0]

    '''load train dataset'''

    data_train = np.load('dataset/zhouqi_train.npz',allow_pickle=True)

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

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(branch_input,trunk_input).to(device)
        loss = criterion(outputs,labels_train).to(device)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 1000 == 0:
            loss_epoches.append(loss.item())
            print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch+1,num_epochs,loss.item()))

    end_time_train = time.time()
    train_time = end_time_train-start_time_train

    print('训练数据共花费时间为',train_time)
    print('loss数据',loss_epoches)

    # # # #save model
    model_path = os.path.join('app/model_temp/deeponet_zhouqi_5s_0.005.pth')
    torch.save(model.state_dict(),model_path)

    '''load test dataset'''
    # start_time = time.time()
    # data_test = np.load('dataset/zhouqi_test.npz', allow_pickle=True)
    # t = data_test['t'].astype(np.float32)
    # len_t = len(t)
    # trunk_input_test = t.reshape(len_t,1)
    # trunk_input_test = trunk_input_test[0:lt:5]
    # trunk_input_test = torch.Tensor(trunk_input_test).to(device)

    # branch_input_test = data_test['X'].astype(np.float32)
    # branch_input_test = branch_input_test[:,0:lt:5]
    # branch_input_test = torch.Tensor(branch_input_test).to(device)

    # labels_test = data_test['Y'].astype(np.float32)
    # labels_test = torch.tensor(labels_test).to(device)
    # labels_test = labels_test[:,0:lt:5]

    '''
    model_path1 ='model_save/deeponet_zhouqi_5s_0.005.pth'
    model.load_state_dict(torch.load(model_path1))
    # model.load_state_dict(torch.load(model_path1,map_location=torch.device('cpu')))
                       
    model.eval()
    with torch.no_grad():
        outputs_test = model(branch_input_test, trunk_input_test).to(device)
    
    end_time = time.time()
    test_time = end_time-start_time
    print('测试数据共花费时间为',test_time)  
    
    
    labels_test_temp = labels_test.cpu().detach().numpy()
    outputs_test_temp = outputs_test.cpu().detach().numpy()
    
    l2_error= l2_relative_error(labels_test_temp,outputs_test_temp)
    mse = mean_squared_error(labels_test_temp,outputs_test_temp)
    R2 = R2(labels_test_temp,outputs_test_temp)
    mae = MAE(labels_test_temp,outputs_test_temp)
    
    print('L2相对误差(l2_relative_error)为：',l2_error)
    print('均方误差(MSE)为',mse)
    print('R2为',R2)
    print('平均绝对误差为',mae)
    '''

    # for i in range(0,50):
    #     plt.show()
    #     x_data = np.arange(0,max_t,0.005)
    #     plt.rcParams['font.sans-serif'] = ['SimHei'] # 推荐使用SimHei字体显示中文
    #     plt.rcParams['axes.unicode_minus'] = False

    #     plt.figure(figsize=(15, 20))

    #     plt.subplot(2, 1, 1)
    #     plt.plot(x_data,branch_input_test[i,:],color = 'r', label = '系统输入')
    #     plt.xlabel('时间(s)')
    #     plt.ylabel('角加速度')
    #     plt.legend(fontsize='18')
    #     plt.grid(True)

    #     plt.subplot(2, 1, 2)
    #     plt.plot(x_data,labels_test[i,:],  label='系统真实响应值')
    #     plt.plot(x_data,outputs_test[i,:],  label='响应值')
    #     plt.plot(x_data,outputs_test1[i,:],  label='阶跃响应值')

    #     plt.xlabel('时间(s)')
    #     plt.ylabel('干扰力矩')
    #     plt.legend(fontsize='18')
    #     plt.grid(True)

    #     plt.show()


if __name__ == '__main__':
    show_deeponet()
