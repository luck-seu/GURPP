import random
import logging
import os
import sys
import numpy as np
import torch
from datetime import datetime

from tqdm import tqdm

from downstream_task import predict_crime, predict_check
from load_graph_data import HeteroGraphData, GURPData
from model.gurp import GURPModel
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
training_key = 'gurp_training'

# Set up random seed
seed = args[training_key]['seed']
torch.manual_seed(seed=seed)
np.random.seed(seed)
random.seed(seed)

# Set up logger
log_folder = args[training_key]['log_folder']
logger_name = args[training_key]['logger_name']
log_level = args[training_key]['log_level']
log_attr = args[training_key]['log_attr']
log_file_name = logger_name + '_' + log_attr + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = get_logger(log_folder, logger_name, log_file_name + '.log', level=log_level)

# Load data and init features
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
hkg = HeteroGraphData(kg_dir=args['data']['kg_dir'],
                      reverse=args['data']['kg_reverse'],
                      export=args['data']['kg_export'],
                      graph_device=device)
logger.info('kg data loaded...')
hkg.init_hetero_graph_features(image=args['data']['image'], flow=args['data']['flow'],
                               feature_dir=args['data']['hg_feature_dir'],
                               node_feats_dim=args['gurp_model']['node_dim'][0],
                               device=device)
logger.info('feature loaded...')

gurp_test_data = GURPData(hkg, if_test=True)
gurp_model = GURPModel(args['gurp_model']).to(device)
optimizer = torch.optim.Adam(gurp_model.parameters(), lr=args[training_key]['lr'], weight_decay=args[training_key]['weight_decay'])

def get_batch_mobility(mobility, batch_size, batch_num):
    mobility_batch = []
    for i in range(batch_num):
        cur_batch_mob = mobility[i*batch_size:(i+1)*batch_size]
        cur_batch_mob = cur_batch_mob.reshape(-1, batch_size, 180)
        mobility_batch.append(cur_batch_mob)
    return mobility_batch

def validate_model(args):
    sp_pos_all, sp_neg_all, attr_pos_all = gurp_test_data.get_all_samples(args['ratio_dict'], args['aug_sel_ratio'],
                                                                    args['seed'])
    logger.info('samples loaded...')

    batch_size = args['batch_size']
    batch_num = 180 // batch_size

    mobility = np.load(args['mobility'])
    mobility = torch.tensor(mobility, dtype=torch.float32).to(device)
    mob_batch = get_batch_mobility(mobility, batch_size, batch_num)

    reg_sub_b, sp_pos_b, sp_neg_b, attr_pos_b, reg_inf, reg_ouf, inf_aug, ouf_aug, img = gurp_test_data.get_batch_samples(
        sp_pos_all, sp_neg_all, attr_pos_all, batch_size, args)

    # model_path = 'model.pth'
    # gurp_model.load_state_dict(torch.load(model_path))
    # gurp_model.eval()

    with torch.no_grad():
        # cur_out_emb = gurp_model(reg_sub_b[0], sp_pos_b[0], sp_neg_b[0], None, reg_inf[0], reg_ouf[0], None, None,
        #                         mob_batch[0], img[0], batch_size, args)[-1].detach().cpu().numpy()

        cur_out_emb = np.load('emb.npy')

        cri_mae, cri_rmse, cri_r2 = predict_crime(cur_out_emb)
        chk_mae, chk_rmse, chk_r2 = predict_check(cur_out_emb)
        logger.info('crime test result, cri_mae: {}, cri_rmse: {}, cri_r2: {}'.format(cri_mae, cri_rmse, cri_r2))
        logger.info('check test result, chk_mae: {}, chk_rmse: {}, chk_r2: {}'.format(chk_mae, chk_rmse, chk_r2))

validate_model(args[training_key])