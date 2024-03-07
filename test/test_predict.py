import configs
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
import torch
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


def predict(model_name='BERT', device='mps'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = model_name, help = 'name of the model')
    parser.add_argument('--save_name', type = str, default=model_name)

    parser.add_argument('--train_prefix', type = str, default = 'train')
    parser.add_argument('--test_prefix', type = str, default = 'dev_test_predict')
    parser.add_argument('--input_theta', type = float, default = -1)
    parser.add_argument('--two_phase', action='store_true')
    # parser.add_argument('--ignore_input_theta', type = float, default = -1)


    args = parser.parse_args()
    model = {
        'LSTM_UP': models.LSTM_UP,
        'LSTM': models.LSTM,
        'BERT': models.BERT,
        'ContextAware': models.ContextAware,
    }

    con = configs.Config(args)
    con.device = torch.device(device)
    #con.load_train_data()
    con.load_predict_data()
    # con.set_train_model()
    pretrain_model_name = model_name
    con.test_predect(model[args.model_name], args.save_name, args.input_theta, args.two_phase, pretrain_model_name)#, args.ignore_input_theta)
