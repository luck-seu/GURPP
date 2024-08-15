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

gurp_data = GURPData(hkg, if_test=False)

gurp_model = GURPModel(args['gurp_model']).to(device)
optimizer = torch.optim.Adam(gurp_model.parameters(), lr=args[training_key]['lr'], weight_decay=args[training_key]['weight_decay'])

def get_batch_mobility(mobility, batch_size, batch_num):
    mobility_batch = []
    for i in range(batch_num):
        cur_batch_mob = mobility[i*batch_size:(i+1)*batch_size]
        cur_batch_mob = cur_batch_mob.reshape(-1, batch_size, 180)
        mobility_batch.append(cur_batch_mob)
    return mobility_batch

def train_model(args):
    epochs = args['epochs']
    batch_size = args['batch_size']
    batch_num = 180 // batch_size
    train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    best_mae_crime = 10000
    best_rmse_crime = 10000
    best_r2_crime = 0
    best_epoch_crime = 0


    sp_pos_all, sp_neg_all = gurp_data.get_all_samples(args['ratio_dict'], args['seed'])
    logger.info('samples loaded...')

    mobility = np.load(args['mobility'])
    mobility = torch.tensor(mobility, dtype=torch.float32).to(device)
    mob_batch = get_batch_mobility(mobility, batch_size, batch_num)
    reg_sub_b, sp_pos_b, sp_neg_b, reg_inf, reg_ouf, img = gurp_data.get_batch_samples(sp_pos_all, sp_neg_all,
                                                                                       batch_size, args)

    for epoch in tqdm(range(args['epochs'])):
        gurp_model.train()
        loss_epoch = []
        for batch in range(batch_num):
            optimizer.zero_grad()
            loss, sp_loss, flow_loss, img_loss, pred_loss, out_emb = gurp_model(reg_sub_b[batch], sp_pos_b[batch],
                                                                                sp_neg_b[batch], reg_inf[batch],
                                                                                reg_ouf[batch], mob_batch[batch],
                                                                                img[batch], batch_size, args)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            logger.info(f"Epoch [{epoch}/{args['epochs']}]"
                        f"\t sp_loss: {sp_loss.item()} "
                        f"\t flow_loss: {flow_loss.item()} "
                        f"\t img_loss: {img_loss.item()}"
                        f"\t pred_loss: {pred_loss.item()}")

        with torch.no_grad():
            cur_out_emb = out_emb.detach().cpu().numpy()
            cri_mae, cri_rmse, cri_r2 = predict_crime(cur_out_emb)
            logger.info('epoch: {}, crime validating result, cri_mae: {}, cri_rmse: {}, cri_r2: {}'.format(epoch, cri_mae,cri_rmse,cri_r2))

            if cri_rmse < best_rmse_crime and cri_mae < best_mae_crime and best_r2_crime < cri_r2:
                best_rmse_crime = cri_rmse
                best_mae_crime = cri_mae
                best_r2_crime = cri_r2
                best_epoch_crime = epoch
                best_emb_crime = out_emb

    torch.save(gurp_model.state_dict(), args['save_model_path'] + f'/gurp_model_epoch_{epochs}_time_{train_time}.pth')
    np.save(args['save_model_path'] + f'/region_emb_epoch_{epochs}_time_{train_time}.npy', cur_out_emb)
    logger.info('best epoch: {}, best crime result, cri_mae: {}, cri_rmse: {}, cri_r2: {}'.format(best_epoch_crime, best_mae_crime, best_rmse_crime, best_r2_crime))

    gurp_model.eval()
    with torch.no_grad():
        cri_mae, cri_rmse, cri_r2 = predict_crime(cur_out_emb)
        chk_mae, chk_rmse, chk_r2 = predict_check(cur_out_emb)
        logger.info('crime final result, cri_mae: {}, cri_rmse: {}, cri_r2: {}'.format(cri_mae, cri_rmse, cri_r2))
        logger.info('check final result, chk_mae: {}, chk_rmse: {}, chk_r2: {}'.format(chk_mae, chk_rmse, chk_r2))
        cri_mae, cri_rmse, cri_r2 = predict_crime(best_emb_crime.detach().cpu().numpy())
        chk_mae, chk_rmse, chk_r2 = predict_check(best_emb_crime.detach().cpu().numpy())
        logger.info('best final result, crime, mae: {}, rmse: {}, r2: {}'.format(cri_mae, cri_rmse, cri_r2))
        logger.info('best final result, check, mae: {}, rmse: {}, r2: {}'.format(chk_mae, chk_rmse, chk_r2))

train_model(args[training_key])