import logging
import os
import sys
import numpy as np
from datetime import datetime

from downstream_task import predict_crime, predict_check
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

args = get_default_arguments()
training_key = 'gurp_training'

log_folder = args[training_key]['log_folder']
logger_name = args[training_key]['logger_name']
log_level = args[training_key]['log_level']
log_attr = args[training_key]['log_attr']
log_file_name = logger_name + '_' + log_attr + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = get_logger(log_folder, logger_name, log_file_name + '.log', level=log_level)

def reproduce_results():
    cur_out_emb = np.load('emb.npy')
    cri_mae, cri_rmse, cri_r2 = predict_crime(cur_out_emb)
    chk_mae, chk_rmse, chk_r2 = predict_check(cur_out_emb)
    logger.info('crime test result, cri_mae: {}, cri_rmse: {}, cri_r2: {}'.format(cri_mae, cri_rmse, cri_r2))
    logger.info('check test result, chk_mae: {}, chk_rmse: {}, chk_r2: {}'.format(chk_mae, chk_rmse, chk_r2))

reproduce_results()