import torch
import torch.nn as nn

from torch.nn import functional as F


class FNN(nn.Module):
    '''Fully-connected neural network.'''


    def __init__(self,layer_sizes):
        super(FNN,self).__init__()
        self.linears = torch.nn.ModuleList()
        for i in range(1,len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i-1],layer_sizes[i]
                )
            )


    def forward(self,inputs):
        x = inputs
        for j,linear in enumerate(self.linears[:-1]):
            x = (
                F.relu(linear(x))
            )
        x = self.linears[-1](x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.branch_layer_size = [600,568,128,64,40]
        self.trunk_layer_size = [1,128,128,128,40]
        self.build_mode()
        self.num_outputs = 1
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)]
        )


    def build_mode(self):
        self.branch_net_output = FNN(self.branch_layer_size)
        self.trunk_net_output = FNN(self.trunk_layer_size)


    def merge_branch_trunk(self,x_func,x_loc,index):
        y = torch.einsum("bi,ni->bn",x_func,x_loc)
        y += self.b[index]
        # y1 = torch.unsqueeze(y, dim=1)
        return y


    def forward(self,input_a,input_b):
        input_a = torch.Tensor(input_a)
        input_a = input_a.unsqueeze(0)
        input_b = torch.Tensor(input_b)
        branch_out = self.branch_net_output(input_a)
        trunk_out = self.trunk_net_output(input_b)
        outputs = self.merge_branch_trunk(branch_out,trunk_out,0)
        outputs = outputs.squeeze(0)
        return outputs
