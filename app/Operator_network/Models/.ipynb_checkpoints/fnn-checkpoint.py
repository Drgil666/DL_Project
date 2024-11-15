''' Fully-connected neural network'''
import torch
import torch.nn as nn
import torch.nn.functional as F

'''初始化方法'''
INITIALIZER_DICT = {
    "Glorot normal":torch.nn.init.xavier_normal_,
    "Glorot uniform":torch.nn.init.xavier_uniform_,
    "He normal":torch.nn.init.kaiming_normal_,
    "He uniform":torch.nn.init.kaiming_uniform_,
    "zeros":torch.nn.init.zeros_,
}


def initializers_get(identifier):
    """Retrieve an initializer by the identifier.

    Args:
        identifier: String that contains the initializer name or an initializer
            function.

    Returns:
        Initializer instance base on the input identifier.
    """
    if isinstance(identifier,str):
        return INITIALIZER_DICT[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret initializer identifier: "+str(identifier))


class FNN(nn.Module):
    def __init__(self,layer_sizes,activation,kernel_initializer):
        super(FNN,self).__init__()
        if isinstance(activation,list):
            if not (len(layer_sizes)-1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
        initializer = initializers_get(kernel_initializer)
        initializer_zero = initializers_get("zeros")
        self.activation = getattr(F,activation)

        self.linears = torch.nn.ModuleList()
        for i in range(1,len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i-1],layer_sizes[i])
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)


    def forward(self,inputs):
        x = inputs
        for j,linear in enumerate(self.linears[:-1]):
            x = (
                self.activation(linear(x))
            )
        x = self.linears[-1](x)
        return x
