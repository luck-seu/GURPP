import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class RW_layer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, max_step=2, size_graph_filter=10, dropout=0.5):
        super(RW_layer, self).__init__()
        self.max_step = max_step
        self.size_graph_filter = size_graph_filter
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.fc_in = torch.nn.Linear(input_dim, hidden_dim)
            self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, hidden_dim, output_dim))
        else:
            self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, input_dim, output_dim))
        self.adj_hidden = Parameter(torch.FloatTensor((size_graph_filter * (size_graph_filter - 1)) // 2, output_dim))

        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)

    def forward(self, adj, features, idxs):
        adj_hidden_norm = torch.zeros(self.size_graph_filter, self.size_graph_filter, self.output_dim).to(device)
        idx = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1)
        adj_hidden_norm[idx[0], idx[1], :] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 0, 1)
        z = self.features_hidden

        x = features
        if self.hidden_dim:
            x = nn.ReLU()(self.fc_in(x))
        x = x[idxs]

        zx = torch.einsum("mcn,abc->ambn", (z, x))
        out = []
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.size_graph_filter, device=device)
                o = torch.einsum("ab,bcd->acd", (eye, z))
                t = torch.einsum("mcn,abc->ambn", (o, x))
            else:
                x = torch.einsum("abc,acd->abd", (adj, x))
                z = torch.einsum("abd,bcd->acd", (adj_hidden_norm, z))
                t = torch.einsum("mcn,abc->ambn", (z, x))
            t = self.dropout(t)
            t = torch.mul(zx, t)
            t = torch.mean(t, dim=[1, 2])
            out.append(t)

        out = sum(out) / len(out)
        return out


class RWPrompt(nn.Module):
    def __init__(self, feature_dim, output_dim=1, hidden_dims=None, size_sub_graph=10,
                 size_graph_filter=None, max_step=1, pretrain_dim=144, dropout_rate=0.5, weight=None, bias=None):

        super(RWPrompt, self).__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32, 16]
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)
        self.num_layers = len(hidden_dims) - 1
        self.size_sub_graph = size_sub_graph

        self.ker_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.ker_layers.append(
                    RW_layer(feature_dim, hidden_dims[1], hidden_dim=hidden_dims[0],
                             max_step=max_step, size_graph_filter=size_graph_filter[0], dropout=dropout_rate))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[1]))
            else:
                self.ker_layers.append(
                    RW_layer(hidden_dims[layer], hidden_dims[layer + 1], hidden_dim=None,
                             max_step=max_step, size_graph_filter=size_graph_filter[layer], dropout=dropout_rate))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[layer + 1]))

        self.linear_prediction = nn.Linear(sum(hidden_dims[1:]) + pretrain_dim, output_dim)

        torch.nn.init.xavier_normal_(self.linear_prediction.weight.data)
        self.linear_prediction.weight.data[:, -pretrain_dim:] = weight
        self.linear_prediction.bias.data = bias

    def forward(self, adj, features, pretrain_emb, idxs):
        hidden_rep = []
        h = features
        for layer in range(self.num_layers):
            h = self.ker_layers[layer](adj, h, idxs)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            hidden_rep.append(h)

        for layer, h in enumerate(hidden_rep):
            if layer == 0:
                h = torch.mean(h.view(-1, self.size_sub_graph, h.shape[1]), dim=1)
                out = h
            else:
                h = torch.mean(h.view(-1, self.size_sub_graph, h.shape[1]), dim=1)
                out = torch.cat([out, h], dim=1)
        out = torch.cat([out, pretrain_emb], dim=1)
        out = self.linear_prediction(out)
        return out