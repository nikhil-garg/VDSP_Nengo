
import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from mnist_vdsp_single import *
from utilis import *
from args_mnist import args as my_args
# from ax import optimize
import pandas as pd
from itertools import product
import time


if __name__ == '__main__':

	args = my_args()
	print(args.__dict__)

	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger(__name__)

	# Fix the seed of all random number generator
	seed = 50
	random.seed(seed)
	np.random.seed(seed)

	parameters = dict(
		digit=[0],
		input_nbr=[60000],
		g_max=[0.3]
		,tau_in = [0.3]
		,tau_out = [0.3]
		, lr = [0.0001]
    )
	param_values = [v for v in parameters.values()]

	now = time.strftime("%Y%m%d-%H%M%S")
	folder = os.getcwd()+"/MNIST_VDSP_explorartion"+now
	os.mkdir(folder)

	for args.digit,args.input_nbr,args.g_max,args.tau_in,args.tau_out,args.lr in product(*param_values):

		args.filename = '-g_max-'+str(args.g_max)+'-tau_in-'+str(args.tau_in)+'-tau_out-'+str(args.tau_out)+'-digit-'+str(args.digit)
		weights = evaluate_mnist_single(args)

		weights = weights[-1] #Only take last weight
		fig, axes = plt.subplots(1,1, figsize=(3,3))

		fig = plt.figure()

		ax1 = fig.add_subplot()

		cax = ax1.matshow(np.reshape(weights[0],(28,28)),interpolation='nearest', vmax=1, vmin=0)

		fig.colorbar(cax)

		plt.tight_layout()    

		fig.savefig(folder+'/weights'+str(args.filename)+'.png')


	logger.info('All done.')