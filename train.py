import torch

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
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


def train(model_name='BERT', device='mps'):
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type = str, default = model_name, help = 'name of the models')
	parser.add_argument('--save_name', type = str, default = model_name)

	parser.add_argument('--train_prefix', type = str, default = 'dev_train')
	parser.add_argument('--test_prefix', type = str, default = 'dev_dev')


	args = parser.parse_args()
	model = {
		'LSTM': models.LSTM,
		'BERT': models.BERT,
		'ContextAware': models.ContextAware,
	}

	con = configs.Config(args)
	con.device = torch.device(device)
	con.set_max_epoch(10)
	con.set_batch_size(24)
	con.load_train_data()
	con.load_test_data()
	# con.set_train_model()
	con.train(model[args.model_name], args.save_name)
