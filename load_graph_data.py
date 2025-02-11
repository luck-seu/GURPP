import os.path
import random

import dgl
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class HeteroGraphData(object):
    def __init__(self, kg_dir, reverse=False, export=False, graph_device='cpu'):
        self.kg_dir = kg_dir
        self.reverse = reverse
        self.ents, self.ent_range, self.rels, self.ent2id, self.rel2id, self.kg_data = self.load_kg(reverse=reverse)
        self.num_ent = len(self.ent2id)
        self.mht_region_ents = self.ents[self.ent_range['region'][0]:self.ent_range['region'][1]]
        self.num_mht_region_ent = len(self.mht_region_ents)
        self.graph_device = graph_device

        if reverse:
            self.num_rel = len(self.rel2id)
        else:
            self.num_rel = len(self.rel2id)

        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        if export:
            print('exporting ent2id.txt and rel2id.txt...')
            file_dir = os.path.dirname(kg_dir)
            if reverse:
                ent2id_file = 'ent2id_reverse.txt'
                rel2id_file = 'rel2id_reverse.txt'
            else:
                ent2id_file = 'ent2id.txt'
                rel2id_file = 'rel2id.txt'
            if not os.path.exists(os.path.join(file_dir, ent2id_file)):
                with open(os.path.join(file_dir, ent2id_file), 'w') as f:
                    for k, v in self.ent2id.items():
                        f.write(k + '\t' + str(v) + '\n')
            else:
                print('ent2id.txt already exists.')
            if not os.path.exists(os.path.join(file_dir, rel2id_file)):
                with open(os.path.join(file_dir, rel2id_file), 'w') as f:
                    for k, v in self.rel2id.items():
                        f.write(k + '\t' + str(v) + '\n')
            else:
                print('rel2id.txt already exists.')

        src = [x[0] for x in self.kg_data]
        dst = [x[2] for x in self.kg_data]
        rels = [x[1] for x in self.kg_data]

        self.g = dgl.graph((src, dst), num_nodes=self.num_ent)
        self.g = self.g.to(graph_device)
        print('num_nodes:', self.g.num_nodes())
        print('homo graph constructed.')

        rel2pairs = {}
        for i, relid in enumerate(rels):
            if self.id2rel[relid] not in rel2pairs.keys():
                rel2pairs[self.id2rel[relid]] = []
            rel2pairs[self.id2rel[relid]].append((src[i], dst[i]))

        rel2pairs_od = {}
        for rel, pairs in rel2pairs.items():
            src = [x[0] for x in pairs]
            dst = [x[1] for x in pairs]
            rel2pairs_od[rel] = (src, dst)

        self.rel2pairs_od = rel2pairs_od

        self.hg = dgl.heterograph({
            ('region', 'NearBy', 'region'): rel2pairs_od['Region_Nearby'],
            ('region', 'HasJunc', 'junction'): (rel2pairs_od['Junction_RegionOf'][1],
                                                [x - self.ent_range['junction'][0] for x in rel2pairs_od['Junction_RegionOf'][0]]),
            ('junction', 'JCateOf', 'junc_cate'): ([x - self.ent_range['junction'][0] for x in rel2pairs_od['Junction_JCateOf'][0]],
                                                   [x - self.ent_range['junc_cate'][0] for x in rel2pairs_od['Junction_JCateOf'][1]]),
            ('region', 'HasRoad', 'road'): (rel2pairs_od['Road_RegionOf'][1],
                                            [x - self.ent_range['road'][0] for x in rel2pairs_od['Road_RegionOf'][0]]),
            ('road', 'RCateOf', 'road_cate'): ([x - self.ent_range['road'][0] for x in rel2pairs_od['Road_RCateOf'][0]],
                                               [x - self.ent_range['road_cate'][0] for x in rel2pairs_od['Road_RCateOf'][1]]),
            ('region', 'HasPoi', 'poi'): (rel2pairs_od['POI_RegionOf'][1],
                                          [x - self.ent_range['poi'][0] for x in rel2pairs_od['POI_RegionOf'][0]]),
            ('poi', 'BrandOf', 'brand'): ([x - self.ent_range['poi'][0] for x in rel2pairs_od['POI_BrandOf'][0]],
                                          [x - self.ent_range['brand'][0] for x in rel2pairs_od['POI_BrandOf'][1]]),
            ('poi', 'Cate1Of', 'cate1'): ([x - self.ent_range['poi'][0] for x in rel2pairs_od['POI_Cate1Of'][0]],
                                          [x - self.ent_range['cate1'][0] for x in rel2pairs_od['POI_Cate1Of'][1]]),
        })

        self.hg = self.hg.to(graph_device)

        print('hetero graph constructed.')
        print(self.hg)

    def load_kg(self, reverse=False):
        facts_str = []
        print('loading knowledge graph...')
        with open(self.kg_dir, 'r') as f:
            f.readline()
            for line in tqdm(f.readlines()):
                x = line.strip().split(',')
                facts_str.append([x[0], x[1], x[2]])

        origin_rels = sorted(list(set([x[1] for x in facts_str])))
        if reverse:
            all_rels = sorted(origin_rels + [x + '_rev' for x in origin_rels])
        else:
            all_rels = sorted(origin_rels)

        all_ents = sorted(list(set([x[0] for x in facts_str] + [x[2] for x in facts_str])))

        mht_region_ents = [x for x in all_ents if x[:4] == 'mhtr']
        mht_region_ents = sorted(mht_region_ents, key=lambda y: int(y[4:]))

        poi_ents = [x[0] for x in facts_str if x[1] == 'POI_Cate1Of']
        poi_ents += [x[0] for x in facts_str if x[1] == 'POI_BrandOf']
        poi_ents += [x[0] for x in facts_str if x[1] == 'POI_RegionOf']
        poi_ents = sorted(list(set(poi_ents)))
        road_ents = [x[0] for x in facts_str if x[1] == 'Road_RCateOf']
        road_ents += [x[0] for x in facts_str if x[1] == 'Road_RegionOf']
        road_ents = sorted(list(set(road_ents)))
        junc_ents = [x[0] for x in facts_str if x[1] == 'Junction_JCateOf']
        junc_ents += [x[0] for x in facts_str if x[1] == 'Junction_RegionOf']
        junc_ents = sorted(list(set(junc_ents)))
        cate1_ents = [x[2] for x in facts_str if x[1] == 'POI_Cate1Of']
        cate1_ents = sorted(list(set(cate1_ents)))
        brand_ents = [x[2] for x in facts_str if x[1] == 'POI_BrandOf']
        brand_ents = sorted(list(set(brand_ents)))
        road_cate_ents = [x[2] for x in facts_str if x[1] == 'Road_RCateOf']
        road_cate_ents = sorted(list(set(road_cate_ents)))
        junc_cate_ents = [x[2] for x in facts_str if x[1] == 'Junction_JCateOf']
        junc_cate_ents = sorted(list(set(junc_cate_ents)))
        streetview_ents = [x[2] for x in facts_str if x[1] == 'Region_StreetViewOf']
        streetview_ents = sorted(list(set(streetview_ents)))

        region_ent2id = dict([(x, i) for i, x in enumerate(mht_region_ents)])
        poi_ent2id = dict([(x, i) for i, x in enumerate(poi_ents)])
        road_ent2id = dict([(x, i) for i, x in enumerate(road_ents)])
        junc_ent2id = dict([(x, i) for i, x in enumerate(junc_ents)])
        cate1_ent2id = dict([(x, i) for i, x in enumerate(cate1_ents)])
        brand_ent2id = dict([(x, i) for i, x in enumerate(brand_ents)])
        road_cate_ent2id = dict([(x, i) for i, x in enumerate(road_cate_ents)])
        junc_cate_ent2id = dict([(x, i) for i, x in enumerate(junc_cate_ents)])
        streetview_ent2id = dict([(x, i) for i, x in enumerate(streetview_ents)])
        ent2id_dicts = {'region': region_ent2id, 'poi': poi_ent2id, 'road': road_ent2id, 'junction': junc_ent2id,
                        'cate1': cate1_ent2id, 'brand': brand_ent2id, 'road_cate': road_cate_ent2id,
                        'junc_cate': junc_cate_ent2id, 'streetview': streetview_ent2id}

        ents = mht_region_ents + poi_ents + road_ents + junc_ents + cate1_ents + brand_ents + road_cate_ents + junc_cate_ents + streetview_ents
        ent_range = {}
        start = 0
        for k, v in ent2id_dicts.items():
            ent_range[k] = (start, start + len(v))
            start += len(v)

        ent2id = dict([(x, i) for i, x in enumerate(ents)])
        rel2id = dict([(x, i) for i, x in enumerate(all_rels)])

        if reverse:
            kg_data = ([[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in facts_str] +
                       [[ent2id[x[2]], rel2id[x[1] + '_rev'], ent2id[x[0]]] for x in facts_str])
        else:
            kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in facts_str]

        print("ent_range: ")
        print(ent_range)
        return ents, ent_range, all_rels, ent2id, rel2id, kg_data

    def init_hetero_graph_features(self, image=None, flow=None, feature_dir=None, node_feats_dim=10, device='cpu'):
        if feature_dir is None:
            print('feature_dir is None, using random initialization...')
            region_feats = torch.randn(self.num_mht_region_ent, node_feats_dim).type(torch.float32).to(device)
            junction_feats = torch.randn(self.hg.num_nodes('junction'), node_feats_dim).type(torch.float32).to(device)
            road_feats = torch.randn(self.hg.num_nodes('road'), node_feats_dim).type(torch.float32).to(device)
            poi_feats = torch.randn(self.hg.num_nodes('poi'), node_feats_dim).type(torch.float32).to(device)
            brand_feats = torch.randn(self.hg.num_nodes('brand'), node_feats_dim).type(torch.float32).to(device)
            cate1_feats = torch.randn(self.hg.num_nodes('cate1'), node_feats_dim).type(torch.float32).to(device)
            junc_cate_feats = torch.randn(self.hg.num_nodes('junc_cate'), node_feats_dim).type(torch.float32).to(device)
            road_cate_feats = torch.randn(self.hg.num_nodes('road_cate'), node_feats_dim).type(torch.float32).to(device)

        else:
            print('feature_dir is not None, using pretrained kg embeddings...')
            ents_emb = np.load(feature_dir[0])
            ent_range = self.ent_range
            region_feats = torch.tensor(ents_emb[ent_range['region'][0]:ent_range['region'][1]], dtype=torch.float32).to(device)
            junction_feats = torch.tensor(ents_emb[ent_range['junction'][0]:ent_range['junction'][1]], dtype=torch.float32).to(device)
            road_feats = torch.tensor(ents_emb[ent_range['road'][0]:ent_range['road'][1]], dtype=torch.float32).to(device)
            poi_feats = torch.tensor(ents_emb[ent_range['poi'][0]:ent_range['poi'][1]], dtype=torch.float32).to(device)
            brand_feats = torch.tensor(ents_emb[ent_range['brand'][0]:ent_range['brand'][1]], dtype=torch.float32).to(device)
            cate1_feats = torch.tensor(ents_emb[ent_range['cate1'][0]:ent_range['cate1'][1]], dtype=torch.float32).to(device)
            junc_cate_feats = torch.tensor(ents_emb[ent_range['junc_cate'][0]:ent_range['junc_cate'][1]], dtype=torch.float32).to(device)
            road_cate_feats = torch.tensor(ents_emb[ent_range['road_cate'][0]:ent_range['road_cate'][1]], dtype=torch.float32).to(device)

        self.hg.nodes['region'].data['f'] = region_feats
        self.hg.nodes['junction'].data['f'] = junction_feats
        self.hg.nodes['road'].data['f'] = road_feats
        self.hg.nodes['poi'].data['f'] = poi_feats
        self.hg.nodes['brand'].data['f'] = brand_feats
        self.hg.nodes['cate1'].data['f'] = cate1_feats
        self.hg.nodes['junc_cate'].data['f'] = junc_cate_feats
        self.hg.nodes['road_cate'].data['f'] = road_cate_feats

        if image is not None:
            si_img_feats = np.load(image)
            si_img_feats = torch.tensor(si_img_feats, dtype=torch.float32).to(device)
            self.hg.nodes['region'].data['si_img'] = si_img_feats
            print('image features loaded.')

        if flow is not None:
            in_flow_feat = np.load(flow[0])
            out_flow_feat = np.load(flow[1])
            scaler = StandardScaler()
            in_flow_scaled = scaler.fit_transform(in_flow_feat)
            out_flow_scaled = scaler.fit_transform(out_flow_feat)
            in_flow = torch.tensor(in_flow_scaled, dtype=torch.float32).to(device)
            out_flow = torch.tensor(out_flow_scaled, dtype=torch.float32).to(device)
            if flow[2] == 'aug':
                in_flow = torch.cat([in_flow for _ in range(6)], dim=1)
                out_flow = torch.cat([out_flow for _ in range(6)], dim=1)
            self.hg.nodes['region'].data['inflow'] = in_flow
            self.hg.nodes['region'].data['outflow'] = out_flow
            print('flow features loaded.')

        return self.hg

    def get_region_sub_all(self):

        region_subgraphs = {}
        region_sub_node_dicts = {}

        for i in range(self.num_mht_region_ent):
            fanout = {'JCateOf':0, 'BrandOf':0, 'Cate1Of':0, 'HasJunc':-1, 'HasPoi':-1, 'HasRoad':-1, 'NearBy':-1,
                      'RCateOf':0}
            sub_graph = dgl.sampling.sample_neighbors(self.hg, {'region': i}, fanout,
                                                      edge_dir='out', copy_ndata=True, copy_edata=True)

            sub_poi_nodes = sub_graph.edges(etype='HasPoi')[1]
            sub_road_nodes = sub_graph.edges(etype='HasRoad')[1]
            sub_junc_nodes = sub_graph.edges(etype='HasJunc')[1]
            sub_region_nodes = sub_graph.edges(etype='NearBy')[1]

            poi_brand_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                            {'JCateOf':0, 'BrandOf':-1, 'Cate1Of':0, 'HasJunc':0,
                                                             'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':0},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)
            poi_cate1_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                            {'JCateOf':0, 'BrandOf':0, 'Cate1Of':-1, 'HasJunc':0,
                                                             'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':0},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)
            junc_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'junction': sub_junc_nodes},
                                                            {'JCateOf':-1, 'BrandOf':0, 'Cate1Of':0, 'HasJunc':0,
                                                             'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':0},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)
            road_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'road': sub_road_nodes},
                                                            {'JCateOf':0, 'BrandOf':0, 'Cate1Of':0, 'HasJunc':0,
                                                             'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':-1},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)

            sub_brand_nodes = poi_brand_graph.edges(etype='BrandOf')[1]
            sub_cate1_nodes = poi_cate1_graph.edges(etype='Cate1Of')[1]
            sub_junc_cate_nodes = junc_cate_graph.edges(etype='JCateOf')[1]
            sub_road_cate_nodes = road_cate_graph.edges(etype='RCateOf')[1]
            region_nodes = torch.cat((torch.tensor([i], dtype=torch.int64).to(self.graph_device), sub_region_nodes), dim=0)

            node_dicts = {'region': region_nodes, 'poi': sub_poi_nodes, 'road': sub_road_nodes, 'junction': sub_junc_nodes,
                          'brand': sub_brand_nodes, 'cate1': sub_cate1_nodes, 'junc_cate': sub_junc_cate_nodes,
                          'road_cate': sub_road_cate_nodes}

            region_sub_node_dicts[i] = node_dicts
            region_subgraphs[i] = dgl.node_subgraph(self.hg, node_dicts)

        print('region subgraphs and node_dicts constructed.')

        return region_subgraphs, region_sub_node_dicts

    def get_region_sub_test_all(self):

        region_subgraphs = {}
        region_sub_node_dicts = {}
        for i in range(self.num_mht_region_ent):
            fanout = {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': -1, 'HasPoi': -1, 'HasRoad': -1,
                      'NearBy': -1, 'RCateOf': 0}
            sub_graph = dgl.sampling.sample_neighbors(self.hg, {'region': i}, fanout,
                                                      edge_dir='out', copy_ndata=True, copy_edata=True)

            sub_poi_nodes = sub_graph.edges(etype='HasPoi')[1]
            sub_road_nodes = sub_graph.edges(etype='HasRoad')[1]
            sub_junc_nodes = sub_graph.edges(etype='HasJunc')[1]
            sub_region_nodes = sub_graph.edges(etype='NearBy')[1]

            seed = 0
            random.seed(seed)
            sub_poi_nodes = random.sample(sub_poi_nodes.tolist(), int(len(sub_poi_nodes) * 0.9))
            sub_road_nodes = random.sample(sub_road_nodes.tolist(), int(len(sub_road_nodes) * 0.9))
            sub_junc_nodes = random.sample(sub_junc_nodes.tolist(), int(len(sub_junc_nodes) * 0.9))

            sub_poi_nodes = torch.tensor(sub_poi_nodes, dtype=torch.long).to(self.graph_device)
            sub_road_nodes = torch.tensor(sub_road_nodes, dtype=torch.long).to(self.graph_device)
            sub_junc_nodes = torch.tensor(sub_junc_nodes, dtype=torch.long).to(self.graph_device)

            poi_brand_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                            {'JCateOf': 0, 'BrandOf': -1, 'Cate1Of': 0, 'HasJunc': 0,
                                                             'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)
            poi_cate1_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                            {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': -1, 'HasJunc': 0,
                                                             'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)
            junc_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'junction': sub_junc_nodes},
                                                            {'JCateOf': -1, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': 0,
                                                             'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)
            road_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'road': sub_road_nodes},
                                                            {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': 0,
                                                             'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': -1},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)

            sub_brand_nodes = poi_brand_graph.edges(etype='BrandOf')[1]
            sub_cate1_nodes = poi_cate1_graph.edges(etype='Cate1Of')[1]
            sub_junc_cate_nodes = junc_cate_graph.edges(etype='JCateOf')[1]
            sub_road_cate_nodes = road_cate_graph.edges(etype='RCateOf')[1]
            region_nodes = torch.cat((torch.tensor([i], dtype=torch.int64).to(self.graph_device), sub_region_nodes),
                                     dim=0)

            node_dicts = {'region': region_nodes, 'poi': sub_poi_nodes, 'road': sub_road_nodes,
                          'junction': sub_junc_nodes, 'brand': sub_brand_nodes,
                          'cate1': sub_cate1_nodes,
                          'junc_cate': sub_junc_cate_nodes, 'road_cate': sub_road_cate_nodes}

            region_sub_node_dicts[i] = node_dicts
            region_subgraphs[i] = dgl.node_subgraph(self.hg, node_dicts)

        print('region subgraphs and node_dicts constructed.')

        return region_subgraphs, region_sub_node_dicts

    def get_region_sub_homo(self, ent_id, hop_k=1, max_node_size=1000):
        in_neighbors = [ent_id]
        for i in range(1, hop_k + 1):
            predecessors = [x for x in in_neighbors for x in self.g.predecessors(x)]
            in_neighbors = list(set(in_neighbors + predecessors))
        out_neighbors = [ent_id]
        for i in range(1, hop_k + 1):
            successors = [x for x in out_neighbors for x in self.g.successors(x)]
            out_neighbors = list(set(out_neighbors + successors))
        sub_nodes = list(set(in_neighbors + out_neighbors))
        if len(sub_nodes) > max_node_size:
            sub_nodes = sub_nodes[:max_node_size]
        else:
            num_virtual_node = max_node_size - len(sub_nodes)
            ori_num_nodes = self.g.num_nodes()
            vnodes_features = torch.zeros(num_virtual_node, self.g.ndata['features'].shape[1]).to(self.graph_device)
            self.g = dgl.add_nodes(self.g, num_virtual_node, {'features': vnodes_features})
            src_nodes = torch.tensor(sub_nodes).to(self.graph_device)
            dst_nodes = torch.tensor([ori_num_nodes + i for i in range(num_virtual_node)]).to(self.graph_device)
            vedges_src = []
            vedges_dst = []
            for i in range(src_nodes.shape[0]):
                for j in range(dst_nodes.shape[0]):
                    vedges_src.append(src_nodes[i])
                    vedges_dst.append(dst_nodes[j])
            self.g = dgl.add_edges(self.g, vedges_src, vedges_dst)
            sub_nodes = sub_nodes + [ori_num_nodes + i for i in range(num_virtual_node)]
        sub_g = dgl.node_subgraph(self.g, torch.tensor(sub_nodes).to(self.graph_device))
        sub_adj = sub_g.adjacency_matrix().to_dense()
        sub_adj = (sub_adj + sub_adj.t()) / 2
        sub_feature = sub_g.ndata['features']
        return sub_adj, sub_feature

class GURPData(object):
    def __init__(self, hkg, if_test=False):
        self.hkg = hkg
        self.hg = hkg.hg
        if if_test:
            self.region_attr_subgraph_all, self.region_sub_nodes_dicts = hkg.get_region_sub_test_all()
        else:
            self.region_attr_subgraph_all, self.region_sub_nodes_dicts = hkg.get_region_sub_all()
        print('subgraph data constructed.')

    def get_all_samples(self, ratio, seed=2024):
        sp_pos_samples = []
        sp_neg_samples = []
        random.seed(seed)
        for i in range(self.hkg.num_mht_region_ent):

            region_nearby_neighbors = self.hg.successors(i, etype='NearBy')
            sp_pos_list = []
            sp_neg_list = []
            for j in range(self.hkg.num_mht_region_ent):
                if j in region_nearby_neighbors:
                    sp_pos_list.append(self.region_attr_subgraph_all[j])
                else:
                    sp_neg_list.append(self.region_attr_subgraph_all[j])
            pos_size = int(len(sp_pos_list) * ratio['sp_pos'])
            if pos_size == 0:
                pos_size = 1
            neg_size = int(len(sp_neg_list) * ratio['sp_neg'])
            if neg_size == 0:
                neg_size = 1
            sp_pos_list = random.sample(sp_pos_list, pos_size)
            sp_neg_list = random.sample(sp_neg_list, neg_size)
            sp_pos_samples.append(sp_pos_list)
            sp_neg_samples.append(sp_neg_list)

        return sp_pos_samples, sp_neg_samples

    def get_batch_samples(self, sp_pos_all, sp_neg_all, batch_size, args):
        reg_sub_batch = []
        sp_pos_batch = []
        sp_neg_batch = []
        reg_inf_batch = []
        reg_outf_batch = []
        reg_img_batch = []
        batch_num = len(sp_pos_all) // batch_size
        sample_size = args['sample_size']
        for i in range(batch_num):
            reg_cur_batch = []
            reg_inf_cur_batch = []
            reg_outf_cur_batch = []
            reg_img_cur_batch = []
            for j in range(batch_size):

                reg_cur_batch.append(self.region_attr_subgraph_all[i * batch_size + j])
                reg_inf = self.hg.nodes['region'].data['inflow'][i * batch_size + j]
                reg_outf = self.hg.nodes['region'].data['outflow'][i * batch_size + j]
                reg_inf_cur_batch.append(reg_inf)
                reg_outf_cur_batch.append(reg_outf)

                reg_img = self.hg.nodes['region'].data['si_img'][i * batch_size + j]
                reg_img_cur_batch.append(reg_img)

            reg_inf_batch.append(torch.stack(reg_inf_cur_batch))
            reg_outf_batch.append(torch.stack(reg_outf_cur_batch))
            reg_sub_batch.append(reg_cur_batch)
            reg_img_batch.append(torch.stack(reg_img_cur_batch))

            if sample_size is None:
                sp_pos_batch.append(sp_pos_all[i * batch_size: (i + 1) * batch_size])
                sp_neg_batch.append(sp_neg_all[i * batch_size: (i + 1) * batch_size])
            else:
                sp_pos_cur_batch = []
                sp_neg_cur_batch = []
                for j in range(batch_size):
                    sp_pos_cur_samples = random.sample(sp_pos_all[i * batch_size + j], sample_size)
                    sp_neg_cur_samples = random.sample(sp_neg_all[i * batch_size + j], sample_size)
                    sp_pos_cur_batch.append(sp_pos_cur_samples)
                    sp_neg_cur_batch.append(sp_neg_cur_samples)
                sp_pos_batch.append(sp_pos_cur_batch)
                sp_neg_batch.append(sp_neg_cur_batch)
        return reg_sub_batch, sp_pos_batch, sp_neg_batch, reg_inf_batch, reg_outf_batch, reg_img_batch