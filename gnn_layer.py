import torch
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
import torch.nn as nn
import torch.nn.functional as F

class GraphLayer(nn.Module):

    def __init__(self,
                 channels_gnn,
                 channels_mlp=None,
                 num_class=None,
                 heads=1,
                 mp_nn="gcn",
                 bias_gnn=True,
                 bias_mlp=True):
        super().__init__()

        channels_gnn.append(num_class)
        
        self.channels_gnn = channels_gnn
        self.mp_nn = mp_nn
        self.bias_gnn = bias_gnn
        self.heads = heads
        self.gnn = nn.ModuleList()
        
        for channel_no in range(len(channels_gnn)-1):
            if self.mp_nn == "gcn":
                self.gnn.append(GCNConv(in_channels=channels_gnn[channel_no], out_channels=channels_gnn[channel_no+1], bias=bias_gnn))
                
            elif self.mp_nn == "gat":
                self.gnn.append(GATConv(in_channels = channels_gnn[channel_no], out_channels=channels_gnn[channel_no+1], heads = self.heads,bias=bias_gnn,concat=False))

            elif self.mp_nn == "sg":
                self.gnn.append(SAGEConv(in_channels=channels_gnn[channel_no], out_channels=channels_gnn[channel_no+1], bias = bias_gnn))
            
            else:
                raise Exception("Check GNN type!")
        # self.channels_mlp = channels_mlp        
        # self.bias_mlp = bias_mlp
        # self.num_class = num_class
        # self.linear = nn.ModuleList()
        # self.channels_mlp.insert(0,channels_gnn[-1])
        # self.channels_mlp.append(num_class)
        # for channel_no in range(0,len(channels_mlp)-1):
        #     self.linear.append(nn.Linear(channels_mlp[channel_no],channels_mlp[channel_no+1], bias=bias_mlp))
        
    def model_parameters(self, model):
        return model.state_dict()

    def weight_update_gnn(self, wgt_add):
        assert len(self.gnn) -1== len(wgt_add), "Match number of GNNs and feature additions"

        # for i in range(1,len(self.channels_gnn)):
        #     self.channels_gnn[i] = self.channels_gnn[i] + wgt_add[i-1]

        for i in range(1,len(self.channels_gnn)-1):
            self.channels_gnn[i] = self.channels_gnn[i] + wgt_add[i-1]

        for i in range(len(self.gnn)):
            model_param = self.model_parameters(self.gnn[i])
            
            if self.mp_nn == "gcn":
                self.gnn[i] = GCNConv(in_channels = self.channels_gnn[i], out_channels = self.channels_gnn[i+1], bias=self.bias_gnn)

            elif self.mp_nn == "gat":
                self.gnn[i] = GATConv(in_channels = self.channels_gnn[i], out_channels = self.channels_gnn[i+1], heads = self.heads, bias=self.bias_gnn, concat=False)

            elif self.mp_nn == "sg":
                self.gnn[i] = SAGEConv(in_channels=self.channels_gnn[i], out_channels=self.channels_gnn[i+1], bias = self.bias_gnn)

            with torch.no_grad():
                if self.mp_nn == "gcn":
                    self.gnn[i].lin.weight[0:model_param["lin.weight"].shape[0] , 0:model_param["lin.weight"].shape[1]] = model_param["lin.weight"]
                    if self.bias_gnn:
                        self.gnn[i].bias[0:model_param["bias"].shape[0]] = model_param["bias"]
                        
                elif self.mp_nn == "gat":
                    self.gnn[i].att_src[0,0:self.heads,0:model_param['att_src'].shape[2]] = model_param['att_src'][0,0:self.heads]
                    self.gnn[i].att_dst[0,0:self.heads,0:model_param['att_dst'].shape[2]] = model_param['att_dst'][0,0:self.heads]
                    self.gnn[i].lin_src.weight[0:model_param["lin_src.weight"].shape[0] , 0:model_param["lin_src.weight"].shape[1]] = model_param["lin_src.weight"]
                    if self.bias_gnn:
                        self.gnn[i].bias[0:model_param["bias"].shape[0]] = model_param["bias"]

                elif self.mp_nn == "sg":
                    self.gnn[i].lin_l.weight[0:model_param["lin_l.weight"].shape[0] , 0:model_param["lin_l.weight"].shape[1]] = model_param["lin_l.weight"]
                    self.gnn[i].lin_r.weight[0:model_param["lin_r.weight"].shape[0] , 0:model_param["lin_r.weight"].shape[1]] = model_param["lin_r.weight"]
                    if self.bias_gnn:
                        self.gnn[i].lin_l.bias[0:model_param["lin_l.bias"].shape[0]] = model_param["lin_l.bias"]
                       
    # def weight_update_mlp(self, wgt_add):
    #     assert len(self.linear)-1 == len(wgt_add), "Match number of Linear layers and node additions"
    #     assert self.channels_mlp[-1] == self.num_class, "MLP output not match class number"

    #     self.channels_mlp[0] = self.channels_gnn[-1] #Change here!
        
    #     for i in range(1,len(self.channels_mlp)-1):
    #         self.channels_mlp[i] = self.channels_mlp[i] + wgt_add[i-1]

    #     for i in range(len(self.linear)):
    #         model_param = self.model_parameters(self.linear[i])
    #         self.linear[i] = nn.Linear(self.channels_mlp[i], self.channels_mlp[i+1], bias=self.bias_mlp)
    #         with torch.no_grad():
    #             self.linear[i].weight[0:model_param["weight"].shape[0] , 0:model_param["weight"].shape[1]] = model_param["weight"]
    #             if self.bias_mlp:
    #                 self.linear[i].bias[0:model_param["bias"].shape[0]] = model_param["bias"]

    def weight_update(self, wgt_gnn=None, wgt_mlp=None):
        self.weight_update_gnn(wgt_gnn)
        # self.weight_update_mlp(wgt_mlp)             
  
    def forward(self,x,edge_index):
        for i in range(len(self.gnn)):
            x = self.gnn[i](x, edge_index)
            x = F.leaky_relu(x)
            x =  F.dropout(x, training=self.training)

        # for i in range(len(self.linear)):
        #     x = self.linear[i](x)
        #     x =  F.dropout(x, training=self.training)
        return x