import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import attr_contrastive_loss_op, mobility_loss
from model.hgt import HGT

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim),
            nn.ReLU()
        )

    def encode(self, feature):
        return self.encoder(feature)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, feature):
        latent = self.encoder(feature)
        fea_new = self.decoder(latent)
        return fea_new, latent


class GURPModel(nn.Module):
    def __init__(self, args):
        super(GURPModel, self).__init__()
        self.seed = args['seed']
        self.node_dict = args['node_dict']
        self.edge_dict = args['edge_dict']
        self.attr_graph_encoder = HGT(node_dict=self.node_dict, edge_dict=self.edge_dict,n_inp=args['node_dim'][0],
                                      n_hid=args['node_dim'][1], n_out=args['node_dim'][2], n_layers=args['n_layers'],
                                      n_heads=args['n_heads'], batch_size=args['batch_size'], agg_method=args['agg_method'],
                                      use_norm=args['use_norm'])
        # flow encoder
        self.flow_encoder = AutoEncoder(in_dim=args['flow_dim'][0], hid_dim=args['flow_dim'][1], out_dim=args['flow_dim'][2])
        # img projector
        self.img_projector = nn.Sequential(
            nn.Linear(512, 144, bias=False),
            nn.ReLU(),
        )
        # fusion layer with batch_norm
        self.fusion = nn.Sequential(
            nn.Linear(args['node_dim'][2] + args['flow_dim'][2] * 2 + 144, args['out_dim']),
            nn.BatchNorm1d(args['out_dim']),
            nn.ReLU()
        )
        # predict in_flow, out_flow, and region
        self.in_flow_predictor = nn.Sequential(
            nn.Linear(args['out_dim'], args['flow_dim'][2]),
            nn.ReLU()
        )
        self.out_flow_predictor = nn.Sequential(
            nn.Linear(args['out_dim'], args['flow_dim'][2]),
            nn.ReLU()
        )
        self.img_predictor = nn.Sequential(
            nn.Linear(args['out_dim'], args['node_dim'][2]),
            nn.ReLU()
        )
        self.region_predictor = nn.Sequential(
            nn.Linear(args['flow_dim'][2], args['node_dim'][2]),
            nn.ReLU()
        )

    def forward(self, region_sub, sp_pos_list, sp_neg_list, region_in_flow, region_out_flow, mobility, img, batch_size, args):
        seed = self.seed
        random.seed(seed)

        region_emb = self.attr_graph_encoder(region_sub)

        sp_pos_sel = []
        sp_neg_sel = []
        for i in range(batch_size):
            sp_pos = sp_pos_list[i]
            sp_neg = sp_neg_list[i]

            if len(sp_pos) == 1:
                sp_pos_sel.append(sp_pos[0])
            else:
                sp_pos_sel.append(sp_pos[torch.randint(0, len(sp_pos), (1,))])
            if len(sp_neg) == 1:
                sp_neg_sel.append(sp_neg[0])
            else:
                sp_neg_sel.append(sp_neg[torch.randint(0, len(sp_neg), (1,))])

        # sp
        sp_pos_emb = self.attr_graph_encoder(sp_pos_sel)
        sp_neg_emb = self.attr_graph_encoder(sp_neg_sel)
        sp_loss_fn = torch.nn.TripletMarginLoss(margin=args['margin'])
        sp_loss = sp_loss_fn(region_emb, sp_pos_emb, sp_neg_emb)

        # flow
        in_flow_emb = self.flow_encoder.encode(region_in_flow)
        out_flow_emb = self.flow_encoder.encode(region_out_flow)
        flow_gen_loss = mobility_loss(out_flow_emb, in_flow_emb, mobility)
        flow_loss = flow_gen_loss

        # image
        img_emb = self.img_projector(img)
        img_loss = attr_contrastive_loss_op(region_emb, img_emb)

        # fusion
        final_region_emb = self.fusion(torch.cat([region_emb, in_flow_emb, out_flow_emb, img_emb], dim=1))

        # pred task
        in_flow_pred = self.in_flow_predictor(final_region_emb) # size 144
        out_flow_pred = self.out_flow_predictor(final_region_emb) # size 144
        img_pred = self.img_predictor(final_region_emb) # size 512
        region_pred = self.region_predictor(final_region_emb) # size 144

        pred_loss = F.mse_loss(torch.cat([in_flow_pred, out_flow_pred, region_pred, img_pred], dim=1),
                               torch.cat([in_flow_emb, out_flow_emb, region_emb, img_emb], dim=1))

        # final loss
        loss = sp_loss + flow_loss + img_loss + pred_loss * args['pred_loss_weight']

        return loss, sp_loss, flow_loss, img_loss, pred_loss, final_region_emb