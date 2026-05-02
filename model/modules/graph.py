from statistics import mode

import math
import numpy as np

import torch
from torch import nn
from torch.nn import Parameter


CONNECTIONS = {0: [1, 8], 1: [0, 2], 2: [1, 3], 3: [2], 4: [5, 8], 5: [4, 6], 6: [5, 7], 7: [6], 8: [0, 4, 9], 9: [8, 10, 12, 17],
               10: [9, 11], 11: [10], 12: [9, 13], 13: [12, 14], 14: [13, 15, 16], 15: [14], 16: [14], 17: [9, 18], 18: [17, 19],
               19: [18, 20, 21], 20: [19], 21: [19]}

class GraphConvolution(nn.Module):


    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        if self.att.shape[0] ==20:
            b,seq,node,f = support.shape
            output = torch.matmul(self.att, support.view(b,node,seq,f))
            output = output.view(b,seq,node,f)
        else :
            output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None,seq=20):
        self.nodes_ = """

        """
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode is undefined"
        self.in_features = dim_in
        self.out_features = dim_out

        self.gc1 = GraphConvolution(dim_in, dim_in, node_n=num_nodes, bias=True)
        self.bn1 = nn.BatchNorm1d(dim_in * num_nodes)
        # self.bn1_1 = nn.BatchNorm1d(dim_in * seq)

        self.gc2 = GraphConvolution(dim_in, dim_in, node_n=num_nodes, bias=True)
        self.bn2 = nn.BatchNorm1d(dim_in * num_nodes)
        # self.bn2_1 = nn.BatchNorm1d(dim_in * seq)

        self.mode = mode
        self.do = nn.Dropout(0.3)
        # self.act_f = nn.Tanh()
        self.act_f = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.gc1(x)
        b, t, n, f = y.shape
        if self.mode  == 'spatial':
            y = self.bn1(y.view(b*t, -1)).view(b,t, n, f)
        # else:
        #     y = self.bn1_1(y.view(b*n, -1)).view(b,t,n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

