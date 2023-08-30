import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3



class FastBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, emb_dim):
        super(FastBatchNorm1d, self).__init__(emb_dim)
        self.mean_scale = ScalePredictor(2*emb_dim, emb_dim // 32, emb_dim)
        self.mean_shift = ShiftPredictor(2*emb_dim, emb_dim // 32, emb_dim)

        self.var_scale = ScalePredictor(2*emb_dim, emb_dim // 32, emb_dim)
        self.var_shift = ShiftPredictor(2*emb_dim, emb_dim // 32, emb_dim)

        self.set_grad_required()
        self.pool = global_mean_pool
        
    def set_grad_required(self):
         for name, p in self.named_parameters():
            if name == 'weight' or name == 'bias':
                p.requires_grad = False

    def get_parameters(self):
        self.parameters_dict = {}
        for name, p in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                self.parameters_dict[name] = p

    def forward(self, x, x_pre, batch, y, flag):
        if flag == True:
            global_x = self.pool(x_pre, batch)
            pos_vec = torch.sum(global_x*y.unsqueeze(-1), dim=0) / torch.sum(y)
            # # normalize pos_vec
            # pos_vec = F.normalize(pos_vec, p=2, dim=-1)
            neg_vec = torch.sum(global_x*(1-y).unsqueeze(-1), dim=0) / torch.sum(1-y)
            # # normalize neg_vec
            # neg_vec = F.normalize(neg_vec, p=2, dim=-1)
            task_vec = torch.cat([pos_vec.unsqueeze(0), neg_vec.unsqueeze(0)], dim=-1)
            task_vec = F.normalize(task_vec, p=2, dim=-1)
            
            
            ################# fast ##################
            self.weight_scale = self.mean_scale(task_vec).t()
            self.weight_shift = self.mean_shift(task_vec).t()
            self.bias_scale = self.var_scale(task_vec).t()
            self.bias_shift = self.var_shift(task_vec).t()

            self.get_parameters()

        self._check_input_dim(x)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight*self.weight_scale[:,0]+self.weight_shift[:,0],
            self.bias*self.bias_scale[:,0]+self.bias_shift[:,0],
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return x

class ScalePredictor(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ScalePredictor, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear_0 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear_1_weight = torch.nn.Parameter(torch.randn(out_dim, hidden_dim) * 0.0001, requires_grad=True)
        self.linear_1_bias = torch.nn.Parameter(torch.ones(out_dim), requires_grad=True)

    def forward(self, x):
        hidden_x = F.leaky_relu(self.linear_0(x), negative_slope=0.0) 
        scale = F.linear(hidden_x, weight=self.linear_1_weight, bias=self.linear_1_bias)
        return scale

class ShiftPredictor(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ShiftPredictor, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear_0 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear_1_weight = torch.nn.Parameter(torch.randn(out_dim, hidden_dim) * 0.0001, requires_grad=True)
        self.linear_1_bias = torch.nn.Parameter(torch.zeros(out_dim), requires_grad=True)

    def forward(self, x):
        hidden_x = F.leaky_relu(self.linear_0(x), negative_slope=0.0)
        scale = F.linear(hidden_x, weight=self.linear_1_weight, bias=self.linear_1_bias)
        return scale

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.emb_dim = emb_dim
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr
        
        # self.m_scale = ScalePredictor(2*emb_dim, emb_dim // 8, emb_dim * 3)
        # self.m_shift = ShiftPredictor(2*emb_dim, emb_dim // 8, emb_dim * 3)

        self.m_scale_0 = ScalePredictor(2*emb_dim, emb_dim // 32, emb_dim * 2 + 1)
        self.m_shift_0 = ShiftPredictor(2*emb_dim, emb_dim // 32, emb_dim * 2 + 1)

        self.m_scale_2 = ScalePredictor(2*emb_dim, emb_dim // 32, emb_dim * 1 + 1)
        self.m_shift_2 = ShiftPredictor(2*emb_dim, emb_dim // 32, emb_dim * 1 + 1)

        self.ee1_scale = ScalePredictor(2*emb_dim, emb_dim // 32, emb_dim)
        self.ee1_shift = ShiftPredictor(2*emb_dim, emb_dim // 32, emb_dim)

        self.ee2_scale = ScalePredictor(2*emb_dim, emb_dim // 32, emb_dim)
        self.ee2_shift = ShiftPredictor(2*emb_dim, emb_dim // 32, emb_dim)                              

        self.set_grad_required()
        self.pool = global_mean_pool
    
    def set_grad_required(self):
         for name, p in self.named_parameters():
            if 'mlp' in name:
                p.requires_grad = False
            if 'edge_embedding' in name:
                p.requires_grad = False

    def get_parameters(self):
        self.parameters_dict = {}
        for name, p in self.named_parameters():
            if p.requires_grad == False:
                self.parameters_dict[name] = p

    def mlp_fun(self, x):
        x = F.linear(x, weight=self.parameters_dict['mlp.0.weight']*self.mlp_0_scale[:-1, :]+self.mlp_0_shift[:-1, :], bias=self.parameters_dict['mlp.0.bias']*self.mlp_0_scale[-1:, 0]+self.mlp_0_shift[-1:, 0])
        x = F.relu(x)
        x = F.linear(x, weight=self.parameters_dict['mlp.2.weight']*self.mlp_1_scale[:-1, :]+self.mlp_1_shift[:-1, :], bias=self.parameters_dict['mlp.2.bias']*self.mlp_1_scale[-1:, 0]+self.mlp_1_shift[-1:, 0])
        return x

    def embedding_fun(self, x, str='1'):
        if str == '1':
            x = F.embedding(x, self.parameters_dict['edge_embedding1.weight']*self.e1_scale+self.e1_shift)
        else:
            x = F.embedding(x, self.parameters_dict['edge_embedding2.weight']*self.e2_scale+self.e2_shift)
        return x

    def forward(self, x, edge_index, edge_attr, batch, y, flag):
        # predict meta-parameters
        if flag == True:
            self.get_parameters()

            global_x = self.pool(x, batch)
            pos_vec = torch.sum(global_x*y.unsqueeze(-1), dim=0) / torch.sum(y)
            # # normalize pos_vec
            # pos_vec = F.normalize(pos_vec, p=2, dim=-1)
            neg_vec = torch.sum(global_x*(1-y).unsqueeze(-1), dim=0) / torch.sum(1-y)
            # # normalize neg_vec
            # neg_vec = F.normalize(neg_vec, p=2, dim=-1)
            task_vec = torch.cat([pos_vec.unsqueeze(0), neg_vec.unsqueeze(0)], dim=-1)
            task_vec = F.normalize(task_vec, p=2, dim=-1)

            
            ################# mlp 1 layer ###################
            # mlp_scale = self.m_scale(task_vec)
            # mlp_shift = self.m_shift(task_vec)
            # self.mlp_0_scale = mlp_scale[:, :2*self.emb_dim].t()
            # self.mlp_0_shift = mlp_shift[:, :2*self.emb_dim].t()
            # self.mlp_1_scale = mlp_scale[:, 2*self.emb_dim:].t()
            # self.mlp_1_shift = mlp_shift[:, 2*self.emb_dim:].t()

            self.mlp_0_scale = self.m_scale_0(task_vec).t()
            self.mlp_0_shift = self.m_shift_0(task_vec).t()

            # ################# embedding ###################
            self.e1_scale = self.ee1_scale(task_vec)
            self.e1_shift = self.ee1_shift(task_vec)
            self.e2_scale = self.ee2_scale(task_vec)
            self.e2_shift = self.ee2_shift(task_vec)

            # ################# mlp 2 layer ###################
            # x_ = F.linear(x, weight=self.parameters_dict['mlp.0.weight']*self.mlp_0_scale[:-1, :]+self.mlp_0_shift[:-1, :], bias=self.parameters_dict['mlp.0.bias']*self.mlp_0_scale[-1:, 0]+self.mlp_0_shift[-1:, 0])
            # x_ = F.relu(x_)
            # global_x = self.pool(x_, batch)
            # pos_vec = torch.sum(global_x*y.unsqueeze(-1), dim=0) / torch.sum(y)
            # # # normalize pos_vec
            # # pos_vec = F.normalize(pos_vec, p=2, dim=-1)
            # neg_vec = torch.sum(global_x*(1-y).unsqueeze(-1), dim=0) / torch.sum(1-y)
            # # # normalize neg_vec
            # # neg_vec = F.normalize(neg_vec, p=2, dim=-1)
            # task_vec = torch.cat([pos_vec.unsqueeze(0), neg_vec.unsqueeze(0)], dim=-1)
            # task_vec = F.normalize(task_vec, p=2, dim=-1)

            self.mlp_1_scale = self.m_scale_2(task_vec).t()
            self.mlp_1_shift = self.m_shift_2(task_vec).t()

        # ee1_scale = self.ee1_scale(global_x)
        # ee1_shift = self.ee1_shift(global_x)

        # ee2_scale = self.ee2_scale(global_x)
        # ee2_shift = self.ee2_shift(global_x)

        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        edge_embeddings = self.embedding_fun(edge_attr[:, 0], str='1') + self.embedding_fun(edge_attr[:, 1], str='2')
        

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp_fun(aggr_out)

class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out

class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)

class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin", batch_norm=True):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.use_batch_norm = batch_norm
        if self.use_batch_norm:
            self.batch_norms = torch.nn.ModuleList()
            for layer in range(num_layer):
                self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
                # self.batch_norms.append(FastBatchNorm1d(emb_dim))
                

    def forward(self, *argv):
        if len(argv) == 3:
            (x, batch, y, flag), edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            (x, batch, y, flag), edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr, batch, y, flag)
            if self.use_batch_norm:
                # h = self.batch_norms[layer](h, h_list[layer], batch, y, flag)
                h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_Encoder(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim,  JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin",batch_norm=True):
        super(GNN_Encoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_workers = 2

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type,batch_norm=batch_norm)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

    def from_pretrained(self, model_file, gpu_id):
        if torch.cuda.is_available():
            res = self.gnn.load_state_dict(torch.load(model_file, map_location='cuda:' + str(gpu_id)), strict=False)
        else:
            res = self.gnn.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=False)
        print(res)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, (batch, y, flag) = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn((x, batch, y, flag), edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)

        return graph_representation, node_representation



