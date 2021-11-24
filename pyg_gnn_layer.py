import torch
from torch_geometric.nn import GCNConv,GATConv,SGConv
import torch.nn as nn
# import torch.nn.functional as F

class GraphLayer(nn.Module):

    def __init__(self,
                 channels_gnn,
                 channels_mlp,
                 heads=2,
                 att_type="gcn",
                 num_class=5,
                 bias_gnn=True,
                 bias_mlp=True):
        super().__init__()
        
        self.channels_gnn = channels_gnn
        self.att_type = att_type
        self.bias_gnn = bias_gnn
        self.heads = heads
        self.gnn = nn.ModuleList()
        
        for channel_no in range(len(channels_gnn)-1):
            if self.att_type == "gcn":
                self.gnn.append(GCNConv(in_channels=channels_gnn[channel_no], out_channels=channels_gnn[channel_no+1], bias=bias_gnn))
                
            elif self.att_type == "gat":
                self.gnn.append(GATConv(in_channels = channels_gnn[channel_no], out_channels=channels_gnn[channel_no+1], heads = self.heads,bias=bias_gnn,concat=False))

            elif self.att_type == "sg":
                self.gnn.append(SGConv(in_channels=channels_gnn[channel_no], out_channels=channels_gnn[channel_no+1], bias = bias_gnn))
            
            else:
                raise Exception("Check GNN type!")

        
        self.channels_mlp = channels_mlp        
        self.bias_mlp = bias_mlp
        self.num_class = num_class
        self.linear = nn.ModuleList()
        self.channels_mlp.insert(0,channels_gnn[-1])
        self.channels_mlp.append(num_class)

        for channel_no in range(0,len(channels_mlp)-1):
            self.linear.append(nn.Linear(channels_mlp[channel_no],channels_mlp[channel_no+1], bias=bias_mlp))
            
    def model_parameters(self, model):
        return model.state_dict()

    def weight_update_gnn(self, wgt_add):
        assert len(self.gnn) == len(wgt_add), "Match number of GNNs and feature additions"

        for i in range(1,len(self.channels_gnn)):
            self.channels_gnn[i] = self.channels_gnn[i] + wgt_add[i-1]

        for i in range(len(self.gnn)):
            model_param = self.model_parameters(self.gnn[i])
            
            if self.att_type == "gcn":
                self.gnn[i] = GCNConv(in_channels = self.channels_gnn[i], out_channels = self.channels_gnn[i+1], bias=self.bias_gnn)

            elif self.att_type == "gat":
                self.gnn.append(GATConv(in_channels = self.channels_gnn[i], out_channels = self.channels_gnn[i+1], heads = self.heads, bias=self.bias_gnn, concat=False))

            elif self.att_type == "sg":
                self.gnn.append(SGConv(in_channels=self.channels_gnn[i], out_channels=self.channels_gnn[i+1], bias = self.bias_gnn))


            with torch.no_grad():
                if self.att_type == "gcn":
                    self.gnn[i].lin.weight[0:model_param["lin.weight"].shape[0] , 0:model_param["lin.weight"].shape[1]] = model_param["lin.weight"]
                    if self.bias_gnn:
                        self.gnn[i].bias[0:model_param["bias"].shape[0]] = model_param["bias"]
                else:
                    raise Exception("Not implemented error")                 #  Implementation needed  
                        
    def weight_update_mlp(self, wgt_add):
        assert len(self.linear)-1 == len(wgt_add), "Match number of Linear layers and node additions"
        self.channels_mlp[0] = self.channels_gnn[-1] #Change here!
        
        for i in range(1,len(self.channels_mlp)-1):
            self.channels_mlp[i] = self.channels_mlp[i] + wgt_add[i-1]

        for i in range(len(self.linear)):
            model_param = self.model_parameters(self.linear[i])
            self.linear[i] = nn.Linear(self.channels_mlp[i], self.channels_mlp[i+1], bias=self.bias_mlp)
            with torch.no_grad():
                self.linear[i].weight[0:model_param["weight"].shape[0] , 0:model_param["weight"].shape[1]] = model_param["weight"]
                if self.bias_mlp:
                    self.linear[i].bias[0:model_param["bias"].shape[0]] = model_param["bias"]

    def weight_update(self, wgt_gnn, wgt_mlp):
        self.weight_update_gnn(wgt_gnn)
        self.weight_update_mlp(wgt_mlp)             
  
    def forward(self,x,edge_index):
        for i in range(len(self.gnn)):
            x = self.gnn[i](x, edge_index)
        for i in range(len(self.linear)):
            x = self.linear[i](x)
        return x

# print("Initial parameters")

# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1],
#                            ], dtype=torch.long).t().contiguous()

# x = torch.tensor([[-1,2], [0,3], [1,5]], dtype=torch.float)

# geo = GraphLayer(channels_gnn = [x.shape[1],3,5], channels_mlp=[8,3])

# # for n,p in geo.named_parameters():
# #     print(n,p,end="\n")

# out_ini = geo(x,edge_index)
# print("Output",out_ini)

# print("--"*120,"\nUpdated parameters")
# wgt_add_gnn = [1,2]
# wgt_add_mlp = [7,3]
# geo.weight_update(wgt_add_gnn,wgt_add_mlp)

# for n,p in geo.named_parameters():
#     print(n,p,end="\n")

# out_upd = geo(x,edge_index)
# print(out_upd)

# print("*"*120)
# print("Update again!")
# wgt_add_gnn = [4,2]
# wgt_add_mlp = [7,5]
# geo.weight_update(wgt_add_gnn,wgt_add_mlp)

# # for n,p in geo.named_parameters():
# #     print(n,p,end="\n")
# print("-"*100)
# out_upd = geo(x,edge_index)
# print(out_upd)