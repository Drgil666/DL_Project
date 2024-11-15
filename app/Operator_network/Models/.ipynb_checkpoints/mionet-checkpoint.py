'''Multiple-input Multiple-output operator network'''

import torch
import torch.nn as nn

from .fnn import FNN


class MIONet(nn.Module):
    def __init__(self,
                 layer_sizes_branch1,
                 layer_sizes_branch2,
                 layer_sizes_branch3,
                 layer_sizes_trunk,
                 activation,
                 kernel_initializer
                 ):
        super(MIONet,self).__init__()
        self.activation = activation

        self.branch_x1 = FNN(
            layer_sizes_branch1,self.activation,kernel_initializer
        )
        self.branch_x2 = FNN(
            layer_sizes_branch2,self.activation,kernel_initializer
        )
        self.branch_x3 = FNN(
            layer_sizes_branch3,self.activation,kernel_initializer
        )

        self.trunk = FNN(layer_sizes_trunk,self.activation,kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))


    def merge_branch_trunk(self,x_func,x_loc):
        y = torch.einsum("bi,ni->bn",x_func,x_loc)
        return y


    def forward(self,input1,input2,input3,t):
        branch_input_1 = input1
        branch_input_2 = input2
        branch_input_3 = input3
        trunk_input = t

        # Trunk net to encode the domain of the output function
        y_loc = self.trunk(trunk_input)

        # Branch net to encode the input function for output1
        y_func_x1 = self.branch_x1(branch_input_1)
        y_func_x2 = self.branch_x2(branch_input_2)
        y_func_x3 = self.branch_x3(branch_input_3)

        x_merger_temp = torch.mul(y_func_x1,y_func_x2)
        x_merger = torch.mul(x_merger_temp,y_func_x3)

        output = torch.einsum("ip,jp->ij",x_merger,y_loc)

        return output
