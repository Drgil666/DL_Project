import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from Models.deeponet import DeepOnet
from metrics import *
from torch import optim


# Step 1: 设置命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="DeepOnet Training Parameters")

    # Training parameters
    parser.add_argument('--num_epochs',type=int,default=27000,help='Number of epochs')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='Learning rate for optimizer')
    parser.add_argument('--batch_size',type=int,default=64,help='Batch size')
    parser.add_argument('--seed',type=int,default=12,help='Random seed for reproducibility')
    parser.add_argument('--branch_layers',type=int,nargs='+',default=[1000,512,256,64],help='branchnet')
    parser.add_argument('--trunk_layers',type=int,nargs='+',default=[1,128,128,64,64],help='trunknet')
    parser.add_argument('--activation',type=str,default="relu",choices=["relu","sigmoid","tanh"],
                        help='Activation function')
    parser.add_argument('--initializer',type=str,default="Glorot normal",help='Weight initializer')

    # Dataset path
    parser.add_argument('--dataset_path',type=str,default='dataset/zhouqi_train.npz',
                        help='Path to the training dataset')

    return parser.parse_args()


# Step 2: 解析命令行参数
args = parse_args()

# Step 3: 使用解析的参数

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)

start_time_train = time.time()

model = DeepOnet(
    args.branch_layers,
    args.trunk_layers,
    args.activation,
    args.initializer
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)

num_epochs = args.num_epochs

dt = 0.001
min_t = 0
max_t = 5
a = np.arange(min_t,max_t,dt)
lt = a.shape[0]

'''load train dataset'''

data_train = np.load(args.dataset_path,allow_pickle=True)

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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
