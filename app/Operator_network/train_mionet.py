import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim

from Models.mionet import MIONet
from metrics import *

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

seed = 12
np.random.seed(seed)
torch.manual_seed(seed)

model = MIONet(
    [2000,1024,256,128,64],
    [2000,1024,256,128,64],
    [2000,1024,256,128,64],
    [1,128,128,64,64],
    "relu",
    "Glorot normal"
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params}")

# criterion = nn.MSELoss()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epochs = 50000

dt = 0.001
min_t = 0
max_t = 10
a = np.arange(min_t,max_t,dt)
lt = a.shape[0]

'''load train dataset'''
data_train = np.load('dataset/3DOF_train.npz',allow_pickle=True)

branch_input = data_train['X']
branch_input = torch.tensor(branch_input,dtype=torch.float32).to(device)
branch_input_1 = branch_input[:,:lt:5,0]
branch_input_2 = branch_input[:,:lt:5,1]
branch_input_3 = branch_input[:,:lt:5,2]

t = data_train['t'].astype(np.float32)
len_t = len(t)
trunk_input = t.reshape(len_t,1)
trunk_input = torch.tensor(trunk_input[:lt:5],dtype=torch.float32).to(device)

label_train = data_train['Y']
label_train = torch.tensor(label_train,dtype=torch.float32).to(device)
label_train_0 = label_train[:,:lt:5,0]
label_train_1 = label_train[:,:lt:5,1]
label_train_2 = label_train[:,:lt:5,2]
#
model.train()

for epoch in range(num_epochs):

    optimizer.zero_grad()
    output = model(branch_input_1,branch_input_2,branch_input_3,trunk_input).to(device)

    loss = criterion(output,label_train_0).to(device)
    loss.backward()
    optimizer.step()
    # sum += loss

    if (epoch+1) % 1000 == 0:
        print('Epoch [{}/{}], Lossx: {:.8f}'.format(epoch+1,num_epochs,loss.item()))

'''save model'''
model_path = os.path.join('model_save/MIOnet_x_10s_0.05.pth')
torch.save(model.state_dict(),model_path)

# '''load train dataset'''
data_train = np.load('dataset/3DOF_test.npz',allow_pickle=True)

branch_input = data_train['X']
branch_input = torch.tensor(branch_input,dtype=torch.float32).to(device)
branch_input_1 = branch_input[:,:lt:5,0]
branch_input_2 = branch_input[:,:lt:5,1]
branch_input_3 = branch_input[:,:lt:5,2]

t = data_train['t'].astype(np.float32)
len_t = len(t)
trunk_input = t.reshape(len_t,1)
trunk_input = torch.tensor(trunk_input[:lt:5],dtype=torch.float32).to(device)

label_train = data_train['Y']
label_train = torch.tensor(label_train,dtype=torch.float32).to(device)
label_train_0 = label_train[:,:lt:5,0]
label_train_1 = label_train[:,:lt:5,1]
label_train_2 = label_train[:,:lt:5,2]

start_time = time.time()

# model_path1 ='model_save\\1.pth'
# model.load_state_dict(torch.load(model_path1,map_location=torch.device('cpu')))
#
model.eval()
with torch.no_grad():
    output_test = model(branch_input_1,branch_input_2,branch_input_3,trunk_input)
#
end_time = time.time()

test_time = end_time-start_time
print('测试33条数据共花费时间为',test_time)

labels_test_temp = label_train_0.cpu().detach().numpy()
outputs_test_temp = output_test.cpu().detach().numpy()

l2_error = l2_relative_error(labels_test_temp,outputs_test_temp)
mse = mean_squared_error(labels_test_temp,outputs_test_temp)
r2 = R2(labels_test_temp,outputs_test_temp)
mae = MAE(labels_test_temp,outputs_test_temp)
print('L2相对误差(l2_relative_error)为：',l2_error)
print('均方误差(MSE)为',mse)
print('R2为',r2)
print('平均绝对误差为',mae)

# #
for i in range(0,33):
    plt.show()
    x_data = np.arange(0,max_t,0.005)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 推荐使用SimHei字体显示中文
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15,25))

    plt.subplot(4,1,1)
    plt.plot(x_data,branch_input_1[i,:],color='r',label='x轴输入')
    plt.xlabel('时间(s)')
    plt.ylabel('角加速度')
    plt.legend(fontsize='18')
    plt.grid(True)

    plt.subplot(4,1,2)
    plt.plot(x_data,branch_input_2[i,:],color='r',label='y轴输入')
    plt.xlabel('时间(s)')
    plt.ylabel('角加速度')
    plt.legend(fontsize='18')
    plt.grid(True)

    plt.subplot(4,1,3)
    plt.plot(x_data,branch_input_3[i,:],color='r',label='z轴输入')
    plt.xlabel('时间(s)')
    plt.ylabel('角加速度')
    plt.legend(fontsize='18')
    plt.grid(True)

    plt.subplot(4,1,4)
    plt.plot(x_data,label_train_0[i,:],label='x')
    # plt.plot(x_data, label_train_1[i, :], label='y')
    # plt.plot(x_data, label_train_2[i, :], label='z')
    plt.plot(x_data,output_test[i,:],label='y轴响应值')

    plt.xlabel('时间(s)')
    plt.ylabel('干扰力矩')
    plt.legend(fontsize='18')
    plt.grid(True)

    plt.show()
