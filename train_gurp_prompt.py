import logging
import os
import sys
import random
from datetime import datetime

import torch
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from torch import optim
from tqdm import tqdm

from downstream_task import compute_metrics
from load_graph_data import HeteroGraphData
from model.prompt import RWPrompt
from gurpp_args import get_default_arguments

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info('Log directory: %s', log_dir)
    return logger

# Set up arguments
args = get_default_arguments()
training_key = 'gurp_prompt_training'

# set up random seed
seed = args[training_key]['seed']
torch.manual_seed(seed=seed)
np.random.seed(seed)
random.seed(seed + 1)

# Set up logger
log_folder = args[training_key]['log_folder']
logger_name = args[training_key]['logger_name']
log_level = args[training_key]['log_level']
log_file_name = logger_name + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = get_logger(log_folder, logger_name, log_file_name + '.log', level=log_level)

# Load data and init features
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
hkg = HeteroGraphData(kg_dir=args['data']['kg_dir'],
                      reverse=args['data']['kg_reverse'],
                      export=args['data']['kg_export'],
                      graph_device=device)
logger.info('kg data loaded...')
mht_region_ents_idx = [hkg.ent2id[x] for x in hkg.mht_region_ents]

pre_train_emb = np.load(args[training_key]['pre_train_region_emb'], allow_pickle=True)
region_features = torch.from_numpy(pre_train_emb).to(device)

check_counts = np.load(args[training_key]['check_counts'], allow_pickle=True).reshape(180, 1)
crime_counts = np.load(args[training_key]['crime_counts'], allow_pickle=True)

ent_features = np.load(args['data']['kg_ent_pretrained_emb'])
ent_features = torch.tensor(ent_features, dtype=torch.float32).to(device)
logger.info('ent_features.shape: {}'.format(ent_features.shape))
hkg.g.ndata['features'] = ent_features

logger.info('features loaded.')

adj_lst = []
feature_lst = []
hop_k = args[training_key]['hop_k']

for ent_id in tqdm(mht_region_ents_idx):
    sub_adj, sub_feature = hkg.get_region_sub_homo(ent_id, hop_k, max_node_size=args[training_key]['sub_max_nsize'])
    adj_lst.append(sub_adj)
    feature_lst.append(sub_feature)
adj_lst = [x.to(device) for x in adj_lst]
feature_lst = [x.to(device) for x in feature_lst]
adj_all = torch.stack(adj_lst, dim=0)
feature_all = torch.stack(feature_lst, dim=0)

def generate_batch_data(feature_all, size_sub_graph, train_index, adj_all, device=device):
    n_nodes = len(train_index) * size_sub_graph
    feature_batch = torch.zeros(n_nodes, feature_all.shape[2])
    idxs_batch = torch.zeros(n_nodes, size_sub_graph)
    adj_batch = torch.zeros(n_nodes, size_sub_graph, size_sub_graph)
    idx = 0
    for i in train_index:
        n = feature_all[i].shape[0]
        feature_batch[idx:idx + n, :] = feature_all[i]
        for j in range(n):
            idxs_batch[idx + j, :] = torch.arange(idx, idx + size_sub_graph)
            adj_batch[idx + j, :, :] = adj_all[i]
        idx += n
    return feature_batch.to(device), adj_batch.to(device), idxs_batch.long().to(device)

def task_prompt(task_counts, epochs=2, size_sub_graph=100, lr=0.001, weight_decay=0.05):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    y_preds = []
    y_truths = []
    index = torch.arange(180)
    logger.info('KFold starts')
    for train_index, test_index in tqdm(kf.split(index)):

        reg = linear_model.Ridge(alpha=1.0)

        X_train = region_features[train_index]
        Y_train = torch.tensor(task_counts[train_index], dtype=torch.float32).to(device)
        Y_test = torch.tensor(task_counts[test_index], dtype=torch.float32).to(device)

        reg.fit(X_train.detach().cpu().numpy(), Y_train.detach().cpu().numpy())
        weight = torch.tensor(reg.coef_, dtype=torch.float32).to(device)
        bias = torch.tensor(reg.intercept_, dtype=torch.float32).to(device)

        feature_batch, adj_batch, idx_batch = generate_batch_data(feature_all, size_sub_graph, train_index, adj_all, device=device)

        feature_test_batch, adj_test_batch, idx_test_batch = generate_batch_data(feature_all, size_sub_graph, test_index,
                                                                 adj_all, device=device)

        prompt = RWPrompt(feature_dim=144, size_sub_graph=size_sub_graph, size_graph_filter=[6,8], max_step=3,
                          pretrain_dim=144, dropout_rate=0.3, weight=weight, bias=bias).to(device)

        optimizer = optim.Adam(prompt.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.MSELoss()

        logger.info('train_index: {}, test_index: {}'.format(train_index, test_index))
        logger.info('training')
        avg_loss = 0
        for i in range(1, epochs+1):
            optimizer.zero_grad()
            y_pred = prompt(adj_batch, feature_batch, region_features[train_index], idx_batch)
            loss = loss_fn(y_pred, Y_train)
            loss.backward(retain_graph=True)
            optimizer.step()
            avg_loss += loss.item()

        with torch.no_grad():
            logger.info('testing')
            y_test = prompt(adj_test_batch, feature_test_batch, region_features[test_index], idx_test_batch)
            y_preds.append(y_test)
            y_truths.append(Y_test)
            mae, rmse, r2 = compute_metrics(y_test.detach().cpu().numpy(), Y_test.detach().cpu().numpy())
            logger.info('testing result, mae: {}, rmse: {}, r2: {}'.format(mae, rmse, r2))

    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save(prompt.state_dict(), os.path.join(args[training_key]['model_save_dir'], 'prompt_epoch_{}_time_{}.pth'.format(epochs, time)))

    y_preds = torch.cat(y_preds)
    y_truths = torch.cat(y_truths)

    mae, rmse, r2 = compute_metrics(np.concatenate(y_preds.detach().cpu().numpy()), np.concatenate(y_truths.detach().cpu().numpy()))
    logger.info('final result, mae: {}, rmse: {}, r2: {}'.format(mae, rmse, r2))
    return mae, rmse, r2


logger.info('******************************task crime starts.***************************')
task_prompt(crime_counts, epochs=200, size_sub_graph=50, lr=0.001, weight_decay=0.0003)
logger.info('******************************task crime done.*****************************')

logger.info('******************************task check starts.***************************')
task_prompt(check_counts, epochs=200, size_sub_graph=50, lr=0.001, weight_decay=0.05)
logger.info('******************************task check done.*****************************')