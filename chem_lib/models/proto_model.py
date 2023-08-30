import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GNN_Encoder
from .relation import MLP,ContextMLP, TaskAwareRelation


class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x


class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()
        self.rel_layer = args.rel_layer
        self.edge_type = args.rel_adj
        self.edge_activation = args.rel_act
        self.gpu_id = args.gpu_id

        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
            print('load pretrained model from', model_file)
            self.mol_encoder.from_pretrained(model_file, self.gpu_id)

        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)

    def to_one_hot(self,class_idx, num_classes=2):
        return torch.eye(num_classes)[class_idx].to(class_idx.device)

    def label2edge(self, label, mask_diag=True):
        # get size
        num_samples = label.size(1)
        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        # compute edge
        edge = torch.eq(label_i, label_j).float().to(label.device)

        # expand
        edge = edge.unsqueeze(1)
        if self.edge_type == 'dist':
            edge = 1 - edge

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(2)).unsqueeze(0).unsqueeze(0).repeat(edge.size(0), 1, 1, 1).to(edge.device)
            edge=edge*diag_mask
        if self.edge_activation == 'softmax':
            edge = edge / edge.sum(-1).unsqueeze(-1)
        return edge

    def relation_forward(self, s_emb, q_emb, s_label=None, q_pred_adj=False,return_adj=False,return_emb=False):
        if not return_emb:
            s_logits, q_logits, adj = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
        else:
            s_logits, q_logits, adj, s_rel_emb, q_rel_emb = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
        if q_pred_adj:
            q_sim = adj[-1][:, 0, -1, :-1]
            q_logits = q_sim @ self.to_one_hot(s_label)
        if not return_emb:
            return s_logits, q_logits, adj
        else:
            return s_logits, q_logits, adj, s_rel_emb, q_rel_emb

    def forward(self, s_data, q_data, s_label=None, q_pred_adj=False):
        s_emb, s_node_emb = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, (s_data.batch, s_data.y, True))
        q_emb, q_node_emb = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, (q_data.batch, q_data.y, False))

        pos_vec = torch.sum(s_emb*s_data.y.unsqueeze(-1), dim=0) / torch.sum(s_data.y)
        neg_vec = torch.sum(s_emb*(1-s_data.y).unsqueeze(-1), dim=0) / torch.sum(1-s_data.y)
        prototypes = torch.cat([neg_vec.unsqueeze(0), pos_vec.unsqueeze(0)], dim=0)

        q_logits = torch.exp(self.scale)*F.cosine_similarity(q_emb.unsqueeze(1).expand(-1, prototypes.shape[0], -1), 
                                                    prototypes.unsqueeze(0).expand(q_emb.shape[0], -1, -1), dim=-1)
        s_logits = torch.exp(self.scale)*F.cosine_similarity(s_emb.unsqueeze(1).expand(-1, prototypes.shape[0], -1), 
                                                    prototypes.unsqueeze(0).expand(s_emb.shape[0], -1, -1), dim=-1)

        return s_logits, q_logits

    def forward_query_loader(self, s_data, q_loader, s_label=None, q_pred_adj=False):
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, (s_data.batch, s_data.y, True))
        pos_vec = torch.sum(s_emb*s_data.y.unsqueeze(-1), dim=0) / torch.sum(s_data.y)
        neg_vec = torch.sum(s_emb*(1-s_data.y).unsqueeze(-1), dim=0) / torch.sum(1-s_data.y)
        prototypes = torch.cat([neg_vec.unsqueeze(0), pos_vec.unsqueeze(0)], dim=0)
        
        y_true_list=[]
        q_logits_list, adj_list = [], []
        for q_data in q_loader:
            q_data = q_data.to(s_emb.device)
            y_true_list.append(q_data.y)
            q_emb,_ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, (q_data.batch, q_data.y, False))
            
            q_logit = torch.exp(self.scale)*F.cosine_similarity(q_emb.unsqueeze(1).expand(-1, prototypes.shape[0], -1), 
                                            prototypes.unsqueeze(0).expand(q_emb.shape[0], -1, -1), dim=-1)

            q_logits_list.append(q_logit)

        q_logits = torch.cat(q_logits_list, 0)
        y_true = torch.cat(y_true_list, 0)
        sup_labels={'support':s_data.y,'query':y_true_list}
        return q_logits, y_true, sup_labels