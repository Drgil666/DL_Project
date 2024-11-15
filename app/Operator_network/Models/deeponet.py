import torch
import torch.nn as nn

from .fnn import FNN


class DeepOnet(nn.Module):
    def __init__(
            self,
            branch_layer_size,
            trunk_layer_size,
            activation,
            kernel_initializer,):
        super(DeepOnet,self).__init__()
        # self.branch_layer_size = [3000,568,128,64,40]
        # self.trunk_layer_size = [1,128,128,128,40]
        self.branch_layer_size = branch_layer_size
        self.trunk_layer_size = trunk_layer_size
        self.normalizer_a = nn.BatchNorm1d(branch_layer_size[0])
        self.normalizer_b = nn.BatchNorm1d(trunk_layer_size[0])
        self.activation = activation
        self.initializer = kernel_initializer

        self.build_mode()
        self.num_outputs = 1
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)]
        )


    def build_mode(self):
        self.branch_net_output = FNN(self.branch_layer_size,self.activation,self.initializer)
        self.trunk_net_output = FNN(self.trunk_layer_size,self.activation,'Glorot normal')


    def merge_branch_trunk(self,x_func,x_loc,index):
        y = torch.einsum("bi,ni->bn",x_func,x_loc)
        y += self.b[index]
        # y1 = torch.unsqueeze(y, dim=1)
        return y


    def forward(self,input_a,input_b):
        input_a = torch.Tensor(input_a)
        input_b = torch.Tensor(input_b)
        input_normalizer_a = self.normalizer_a(input_a)
        input_normalizer_b = self.normalizer_b(input_b)
        branch_out = self.branch_net_output(input_normalizer_a)
        trunk_out = self.trunk_net_output(input_normalizer_b)
        outputs = self.merge_branch_trunk(branch_out,trunk_out,0)
        return outputs
