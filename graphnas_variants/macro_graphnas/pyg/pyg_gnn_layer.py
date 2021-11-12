import torch
import torch.nn.functional as F
# from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.utils import remove_self_loops, add_self_loops, add_remaining_self_loops, softmax
# from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv,GATConv,SGConv
import torch.nn as nn

# from graphnas_variants.macro_graphnas.pyg.message_passing import MessagePassing
# from torch_geometric.nn import MessagePassing

#########################         Not working ########### Error in mat-mul
class GeoLayer(nn.Module):

    def __init__(self,
                 channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 att_type="gcn",
                 bias=True,):
        super().__init__()
        
        self.channels = channels
        # self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.att_type = att_type
        self.bias = bias

        self.gnn_list = nn.ModuleList()

        for channel_no in range(len(self.channels)-1):
            if self.att_type == "gcn":
                gnn = GCNConv(in_channels=self.channels[channel_no], out_channels=self.channels[channel_no+1], bias=self.bias)
        
            if self.att_type == "gat":
                gnn = GATConv(in_channels = self.channels[channel_no], out_channels = self.channels[channel_no+1], heads = self.heads,bias=self.bias)

            if self.att_type == "sg":
                gnn = SGConv(in_channels=self.channels[channel_no], out_channels=self.channels[channel_no+1], bias = self.bias)

            self.gnn_list.append(gnn)
    
    def model_parameters(self, model):
        return model.state_dict()

    def weight_update(self, wgt_add):
        
        for i in range(len(self.gnn_list)):
            model_param = self.model_parameters(self.gnn_list[i])
            
            if i==0:
                if self.att_type == "gcn":
                    self.gnn_list[i] = GCNConv(in_channels = self.channels[i], out_channels = self.channels[i+1]+wgt_add[i], bias=self.bias)
        
                if self.att_type == "gat":
                    self.gnn_list[i] = GATConv(in_channels = self.channels[i], out_channels = self.channels[i+1]+wgt_add[i], heads = self.heads, bias=self.bias)

                if self.att_type == "sg":
                    self.gnn_list[i] = SGConv(in_channels = self.channels[i], out_channels = self.channels[i+1]+wgt_add[i], bias = self.bias)

            else:
                if self.att_type == "gcn":
                    self.gnn_list[i] = GCNConv(in_channels = self.channels[i]+wgt_add[i-1], out_channels = self.channels[i+1]+wgt_add[i], bias=self.bias)
        
                if self.att_type == "gat":
                    self.gnn_list[i] = GATConv(in_channels = self.channels[i]+wgt_add[i-1], out_channels = self.channels[i+1]+wgt_add[i], heads = self.heads, bias=self.bias)

                if self.att_type == "sg":
                    self.gnn_list[i] = SGConv(in_channels = self.channels[i]+wgt_add[i-1], out_channels = self.channels[i+1]+wgt_add[i], bias = self.bias)
            
            with torch.no_grad():
                if self.att_type == "gcn":
                    self.gnn_list[i].lin.weight = nn.Parameter(torch.cat((model_param["lin.weight"],torch.randn(wgt_add[i],self.channels[i])),0))
                    if self.bias:
                        self.gnn_list[i].bias = nn.Parameter(torch.cat((model_param["bias"],torch.zeros(wgt_add[i])),0))
                        

    def forward(self,x,edge_index):
        for i in range(len(self.gnn_list)):
            x = self.gnn_list[i](x, edge_index)
        return x
        
geo = GeoLayer(channels = [2,3,4])

print("Initial parameters")

for n,p in geo.named_parameters():
    print(n,p.shape,end="\n\n")

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long).t().contiguous()

x = torch.tensor([[-1,2], [0,3], [1,5]], dtype=torch.float)
# print(geo(x,edge_index))
print("*"*120)
# mod["lin.weight"]
hid = [1,1]
geo.weight_update(hid)
# geo(data.x, data.edge_index)

print("Updated parameters")
print(geo(x,edge_index))
for n,p in geo.named_parameters():
    print(n,p.shape,end="\n\n")
print("-"*100)

    
    # @staticmethod
    # def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
    #     if edge_weight is None:
    #         edge_weight = torch.ones((edge_index.size(1), ),
    #                                  dtype=dtype,
    #                                  device=edge_index.device)

    #     fill_value = 1.0
    #     edge_index, edge_weight = add_remaining_self_loops(
    #         edge_index, edge_weight,fill_value, num_nodes)

    #     row, col = edge_index
    #     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    #     deg_inv_sqrt = deg.pow(-0.5)
    #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    #     return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # def reset_parameters(self):
    #     glorot(self.weight)
    #     glorot(self.att)
    #     zeros(self.bias)

    #     if self.att_type in ["generalized_linear"]:
    #         glorot(self.general_att_layer.weight)

    #     if self.pool_dim != 0:
    #         for layer in self.pool_layer:
    #             glorot(layer.weight)
    #             zeros(layer.bias)

    # def forward(self, x, edge_index):
    #     """"""
    #     edge_index, _ = remove_self_loops(edge_index)
    #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    #     # prepare
    #     x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
    #     return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    # def message(self, x_i, x_j, edge_index, num_nodes):

    #     if self.att_type == "const":
    #         if self.training and self.dropout > 0:
    #             x_j = F.dropout(x_j, p=self.dropout, training=True)
    #         neighbor = x_j
    #     elif self.att_type == "gcn":
    #         if self.gcn_weight is None or self.gcn_weight.size(0) != x_j.size(0):  # 对于不同的图gcn_weight需要重新计算
    #             _, norm = self.norm(edge_index, num_nodes, None)
    #             self.gcn_weight = norm
    #         neighbor = self.gcn_weight.view(-1, 1, 1) * x_j
    #     else:
    #         # Compute attention coefficients.
    #         alpha = self.apply_attention(edge_index, num_nodes, x_i, x_j)
    #         alpha = softmax(alpha, edge_index[0], num_nodes)
    #         # Sample attention coefficients stochastically.
    #         if self.training and self.dropout > 0:
    #             alpha = F.dropout(alpha, p=self.dropout, training=True)

    #         neighbor = x_j * alpha.view(-1, self.heads, 1)
    #     if self.pool_dim > 0:
    #         for layer in self.pool_layer:
    #             neighbor = layer(neighbor)
    #     return neighbor

    # def apply_attention(self, edge_index, num_nodes, x_i, x_j):
    #     if self.att_type == "gat":
    #         alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
    #         alpha = F.leaky_relu(alpha, self.negative_slope)

    #     elif self.att_type == "gat_sym":
    #         wl = self.att[:, :, :self.out_channels]  # weight left
    #         wr = self.att[:, :, self.out_channels:]  # weight right
    #         alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
    #         alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
    #         alpha = F.leaky_relu(alpha, self.negative_slope) + F.leaky_relu(alpha_2, self.negative_slope)

    #     elif self.att_type == "linear":
    #         wl = self.att[:, :, :self.out_channels]  # weight left
    #         wr = self.att[:, :, self.out_channels:]  # weight right
    #         al = x_j * wl
    #         ar = x_j * wr
    #         alpha = al.sum(dim=-1) + ar.sum(dim=-1)
    #         alpha = torch.tanh(alpha)
    #     elif self.att_type == "cos":
    #         wl = self.att[:, :, :self.out_channels]  # weight left
    #         wr = self.att[:, :, self.out_channels:]  # weight right
    #         alpha = x_i * wl * x_j * wr
    #         alpha = alpha.sum(dim=-1)

    #     elif self.att_type == "generalized_linear":
    #         wl = self.att[:, :, :self.out_channels]  # weight left
    #         wr = self.att[:, :, self.out_channels:]  # weight right
    #         al = x_i * wl
    #         ar = x_j * wr
    #         alpha = al + ar
    #         alpha = torch.tanh(alpha)
    #         alpha = self.general_att_layer(alpha)
    #     else:
    #         raise Exception("Wrong attention type:", self.att_type)
    #     return alpha

    # def update(self, aggr_out):
    #     if self.concat is True:
    #         aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
    #     else:
    #         aggr_out = aggr_out.mean(dim=1)

    #     if self.bias is not None:
    #         aggr_out = aggr_out + self.bias
    #     return aggr_out

    # def __repr__(self):
    #     return '{}({}, {}, heads={})'.format(self.__class__.__name__,
    #                                          self.in_channels,
    #                                          self.out_channels, self.heads)

    # def get_param_dict(self):
    #     params = {}
    #     key = f"{self.att_type}_{self.agg_type}_{self.in_channels}_{self.out_channels}_{self.heads}"
    #     weight_key = key + "_weight"
    #     att_key = key + "_att"
    #     agg_key = key + "_agg"
    #     bais_key = key + "_bais"

    #     params[weight_key] = self.weight
    #     params[att_key] = self.att
    #     params[bais_key] = self.bias
    #     if hasattr(self, "pool_layer"):
    #         params[agg_key] = self.pool_layer.state_dict()

    #     return params

    # def load_param(self, params):
    #     key = f"{self.att_type}_{self.agg_type}_{self.in_channels}_{self.out_channels}_{self.heads}"
    #     weight_key = key + "_weight"
    #     att_key = key + "_att"
    #     agg_key = key + "_agg"
    #     bais_key = key + "_bais"

    #     if weight_key in params:
    #         self.weight = params[weight_key]

    #     if att_key in params:
    #         self.att = params[att_key]

    #     if bais_key in params:
    #         self.bias = params[bais_key]

    #     if agg_key in params and hasattr(self, "pool_layer"):
    #         self.pool_layer.load_state_dict(params[agg_key])