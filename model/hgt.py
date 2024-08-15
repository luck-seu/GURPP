import math

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax


class HGTLayer(nn.Module):
    """One layer of HGT."""
    def __init__(self, in_dim, out_dim, node_dict, edge_dict, n_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = (
                    sub_graph.edata.pop("t").sum(-1)
                    * relation_pri
                    / self.sqrt_dk
                )
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "t", "m"),
                        fn.sum("m", "t"),
                    )
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes:
                """
                Step 3: Target-specific Aggregation
                x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    def __init__(self,
        node_dict,
        edge_dict,
        n_inp, n_hid, n_out,
        n_layers,
        n_heads,
        batch_size,
        agg_method="concat", use_norm=True,
    ):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_norm = use_norm
        self.batch_size = batch_size
        self.agg_method = agg_method
        self.gcs = nn.ModuleList()
        self.adapt_ws = nn.ModuleList()
        for _ in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm))
        self.gcs_list = nn.ModuleList()
        for _ in range(batch_size):
            self.gcs_list.append(self.gcs)
        self.out = nn.ModuleList()
        for _ in range(len(node_dict)):
            self.out.append(nn.Linear(n_hid, n_out))
        self.out_ntype_to_one = nn.Linear(n_out*len(self.node_dict), n_out)


    def forward(self, GList):
        batched_graph = dgl.batch(GList)
        h_batch = {}
        for ntype in batched_graph.ntypes:
            n_id = self.node_dict[ntype]
            h_batch[ntype] = F.gelu(self.adapt_ws[n_id](batched_graph.nodes[ntype].data["f"]))
        for i in range(self.n_layers):
            h_batch = self.gcs[i](batched_graph, h_batch)

        h = [{} for _ in range(len(GList))]
        node_count = {ntype: 0 for ntype in batched_graph.ntypes}

        for index, G in enumerate(GList):
            for ntype in G.ntypes:
                num_nodes = G.number_of_nodes(ntype)
                h[index][ntype] = h_batch[ntype][node_count[ntype]:node_count[ntype] + num_nodes]
                node_count[ntype] += num_nodes

        out_embedding_list = []
        for index in range(len(GList)):

            key_list = list(self.node_dict.keys())
            if len(self.node_dict) < 2:
                out_embedding_list.append(self.out[0](h[key_list[0]]))
            else:
                out_embedding_list_4_1_graph = []
                for i in range(len(self.node_dict)):
                    out_embedding_list_4_1_graph.append(self.out[i](h[index][key_list[i]]))
                    out_embedding_list_4_1_graph[i] = torch.sum(out_embedding_list_4_1_graph[i], dim=0, keepdim=True)

                out_embedding_list_4_1_graph = torch.stack(out_embedding_list_4_1_graph, dim=0).reshape(-1)
                out_embedding_list.append(self.out_ntype_to_one(out_embedding_list_4_1_graph))

        out_embedding_list = torch.stack(out_embedding_list, dim=0).reshape(self.batch_size, -1)
        return out_embedding_list