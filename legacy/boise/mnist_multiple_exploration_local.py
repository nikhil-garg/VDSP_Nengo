
import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from mnist_vdsp_multiple_local import *
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
	pwd = os.getcwd()
	df = pd.DataFrame({	"vprog":[],
						"vth":[],
						"input_nbr":[],
						"tau_in" :[],
						"tau_out":[],
                        "lr":[],
                        "iterations":[],
                        "presentation_time":[],
                        "dt":[],
                        "n_neurons":[],
                        "inhibition_time":[],
                        "accuracy":[]
                         })

	if args.log_file_path is None:
		log_dir = pwd+'/log_dir/'
	else : 
		log_dir = args.log_file_path
		df.to_csv(log_dir+'test.csv', index=False)


	parameters = dict(
		vprog = [-0.75]
		,input_nbr=[10,60000]
		,tau_in = [0.06]
		,tau_out = [0.06]
		, lr = [0.1]
		,iterations=[1]
		, presentation_time = [0.35]
		, dt = [0.005]
		, n_neurons = [30]
		, inhibition_time = [10]
    )
	param_values = [v for v in parameters.values()]

	now = time.strftime("%Y%m%d-%H%M%S")
	folder = os.getcwd()+"/MNIST_VDSP_explorartion"+now
	os.mkdir(folder)

	for args.vprog,args.input_nbr,args.tau_in,args.tau_out,args.lr,args.iterations,args.presentation_time, args.dt,args.n_neurons,args.inhibition_time in product(*param_values):

		# args.filename = 'vprog-'+str(args.vprog)+'-g_max-'+str(args.g_max)+'-tau_in-'+str(args.tau_in)+'-tau_out-'+str(args.tau_out)+'-lr-'+str(args.lr)+'-presentation_time-'+str(args.presentation_time)
		

		timestr = time.strftime("%Y%m%d-%H%M%S")
		log_file_name = 'accuracy_log'+str(timestr)+'.csv'
		pwd = os.getcwd()


		accuracy, weights = evaluate_mnist_multiple_local(args)



		df = df.append({ "vprog":args.vprog,
						 "vth":args.vthp,
						 "input_nbr":args.input_nbr,
						 "tau_in":args.tau_in,
						 "tau_out": args.tau_out,
						 "lr": args.lr,
						 "iterations":args.iterations,
		                 "presentation_time":args.presentation_time,
		                 "dt":args.dt,
		                 "n_neurons":args.n_neurons,
		                 "inhibition_time":args.inhibition_time,
		                 "accuracy":accuracy
		                 },ignore_index=True)
		

		plot = True
		if plot : 	
			print('accuracy', accuracy)
			print(args.filename)
			# weights = weights[-1]#Taking only the last weight for plotting
			plt.axis('off')
			columns = int(args.n_neurons/6)

			fig, axes = plt.subplots(int(args.n_neurons/columns), int(columns), figsize=(25,30))
			
			for i in range(0,(args.n_neurons)):
				axes[int(i/columns)][int(i%columns)].set_axis_off()
				axes[int(i/columns)][int(i%columns)].matshow(np.reshape(weights[i],(28,28)),interpolation='nearest', vmax=1, vmin=0)

			plt.tight_layout()    

	   
			# fig, axes = plt.subplots(1,1, figsize=(3,3))
			# fig = plt.figure()
			# ax1 = fig.add_subplot()
			# cax = ax1.matshow(np.reshape(weights[0],(28,28)),interpolation='nearest', vmax=1, vmin=0)
			# fig.colorbar(cax)
			# plt.tight_layout()    

			fig.savefig(folder+'/weights'+str(args.filename)+'.svg', bbox_inches='tight',pad_inches = 0)
			fig.savefig(folder+'/weights'+str(args.filename)+'.pdf', bbox_inches='tight',pad_inches = 0)
			plt.close()


			# plt.figure(figsize=(12,10))

			# plt.subplot(2, 1, 1)
			# plt.title('Input neurons')
			# rasterplot(time_points, p_input_layer)
			# plt.xlabel("Time [s]")
			# plt.ylabel("Neuron index")

			# plt.subplot(2, 1, 2)
			# plt.title('Output neurons')
			# rasterplot(time_points, p_layer_1)
			# plt.xlabel("Time [s]")
			# plt.ylabel("Neuron index")

			# plt.tight_layout()

			# plt.savefig(folder+'/raster'+str(args.filename)+'.png')
		timestr = time.strftime("%Y%m%d-%H%M%S")
		log_file_name = 'accuracy_log'+'.csv'
		pwd = os.getcwd()

		if args.log_file_path is None:
			log_dir = pwd+'/log_dir/'
		else : 
			log_dir = args.log_file_path
		df.to_csv(log_dir+log_file_name, index=False)

	df.to_csv(log_file_name, index=False)


	logger.info('All done.')