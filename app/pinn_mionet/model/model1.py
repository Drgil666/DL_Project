"力矩和位置的输出使用了共享特征提取网络，也就是只有一个分支的Mionet"

import torch
import torch.nn as nn
from .mionet1 import MIONet


class PINN_MIONet(nn.Module):
    def __init__(self,config):
        super(PINN_MIONet,self).__init__()

        self.config = config
        self.Td_mionet = MIONet(
            [3500,1024,256,128,64],
            [3500,1024,256,128,64],
            [3500,1024,256,128,64],
            [1,128,128,64,64],
            "tanh",
            "Glorot normal"
        )
        # self.Td_mionet = MIONet(
        #     [2000, 1024, 256, 128, 64],
        #     [2000, 1024, 256, 128, 64],
        #     [2000, 1024, 256, 128, 64],
        #     [1, 128, 128, 64, 64],
        #     "tanh",
        #     "Glorot normal"
        # )

        self.Td_x = config.Td_x_hidden_dim
        self.Td_y = config.Td_y_hidden_dim
        self.Td_z = config.Td_z_hidden_dim

        self.Yita_x = config.Yita_one_hidden_dim
        self.Yita_y = config.Yita_two_hidden_dim
        self.Yita_z = config.Yita_three_hidden_dim

        self.proj1_Td_x = nn.Linear(self.Td_x[0],self.Td_x[1])
        self.proj2_Td_x = nn.Linear(self.Td_x[1],self.Td_x[2])
        self.active = torch.tanh

        self.proj1_Td_y = nn.Linear(self.Td_y[0],self.Td_y[1])
        self.proj2_Td_y = nn.Linear(self.Td_y[1],self.Td_y[2])

        self.proj1_Td_z = nn.Linear(self.Td_z[0],self.Td_z[1])
        self.proj2_Td_z = nn.Linear(self.Td_z[1],self.Td_z[2])

        self.proj1_Yita_x = nn.Linear(self.Yita_x[0],self.Yita_x[1])
        self.proj2_Yita_x = nn.Linear(self.Yita_x[1],self.Yita_x[2])

        self.proj1_Yita_y = nn.Linear(self.Yita_y[0],self.Yita_y[1])
        self.proj2_Yita_y = nn.Linear(self.Yita_y[1],self.Yita_y[2])

        self.proj1_Yita_z = nn.Linear(self.Yita_z[0],self.Yita_z[1])
        self.proj2_Yita_z = nn.Linear(self.Yita_z[1],self.Yita_z[2])


    def forward(self,branch_input1,branch_input2,branch_input3,trunk_input):
        # print("trunk_input in Yita_mionet:", trunk_input.requires_grad)
        # MIOnet_output = self.Td_mionet(branch_input1, branch_input2, branch_input3, trunk_input)
        Yita_MIOnet_output = self.Td_mionet(branch_input1,branch_input2,branch_input3,trunk_input)

        Yita_x1 = self.proj1_Yita_x(Yita_MIOnet_output)
        Yita_x1_active = self.active(Yita_x1)
        Yita_x2 = self.proj2_Td_x(Yita_x1_active)

        Yita_y1 = self.proj1_Yita_y(Yita_MIOnet_output)
        Yita_y1_active = self.active(Yita_y1)
        Yita_y2 = self.proj2_Yita_y(Yita_y1_active)

        Yita_z1 = self.proj1_Yita_z(Yita_MIOnet_output)
        Yita_z1_active = self.active(Yita_z1)
        Yita_z2 = self.proj2_Yita_z(Yita_z1_active)

        return Yita_x2,Yita_y2,Yita_z2
