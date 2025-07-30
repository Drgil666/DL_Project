import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt


# 定义 Positional Encoding 用于加入位置信息
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=28*28):
        super(PositionalEncoding,self).__init__()
        self.encoding = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(torch.tensor(10000.0))/d_model))
        self.encoding[:,0::2] = torch.sin(position*div_term)
        self.encoding[:,1::2] = torch.cos(position*div_term)
        self.encoding = self.encoding.unsqueeze(0)


    def forward(self,x):
        return x+self.encoding[:,:x.size(1)].to(x.device)


# 定义 Transformer 模型
class TransformerClassifier(nn.Module):
    def __init__(self,embedding_dim,num_heads,num_layers,hidden_dim,num_classes,dropout_rate):
        super(TransformerClassifier,self).__init__()
        self.embedding = nn.Linear(28*28,embedding_dim)  # 将28x28的图像像素展平并嵌入embedding_dim维度
        self.position_encoding = PositionalEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,nhead=num_heads,dim_feedforward=hidden_dim,
                                                   dropout=dropout_rate,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim,num_classes)  # 分类器


    def forward(self,x):
        # 将输入图像从批次维度和图像维度（假设为batch_size x 1 x 28 x 28）展平成二维张量（batch_size x 28*28），即四维变成二维
        x = x.view(x.size(0),-1)  # x shape: [batch_size, 28*28]
        # x形状为[batch_size, 28 * 28]，经过self.embedding(x)
        # 后变为[batch_size, embedding_dim]。使用.unsqueeze(1)
        # 后，张量的形状变为[batch_size, 1, embedding_dim]
        x = self.embedding(x).unsqueeze(1)  # 添加一个长度为1的序列维度

        # 加入位置编码
        x = self.position_encoding(x)

        # 通过Transformer编码层
        x = self.transformer_encoder(x)

        # 取出最后一个编码后的特征向量
        x = x.mean(dim=1)  # 平均池化得到一个全局表示

        # 分类
        return self.fc(x)
