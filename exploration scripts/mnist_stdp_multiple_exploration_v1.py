import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from mnist_stdp_multiple_baseline import *
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


	df = pd.DataFrame({	
						"amp_neuron":[],
						"input_nbr":[],
						"tau_in" :[],
						"tau_out":[],                       
                       "alpha_p":[],
                       "alpha_n":[],
                       "beta_p":[],
                       "beta_n":[],
                       "tau_pre":[],
                        "tau_post":[],
                        "iterations":[],
                        "presentation_time":[],
                        "pause_time":[],
                        "dt":[],
                        "n_neurons":[],
                        "inhibition_time":[],
                        "tau_ref_in":[],
                        "tau_ref_out":[],
                        "inc_n":[],
                        "tau_n":[],
                        "synapse_layer_1":[],               
                        "gain_in":[],
                        "gain_out":[],
                        "accuracy":[],
                        "accuracy_2":[]
                         })

	if args.log_file_path is None:
		log_dir = pwd+'/log_dir/'
	else : 
		log_dir = args.log_file_path
		df.to_csv(log_dir+'test.csv', index=False)

	parameters = dict(
		 amp_neuron=[1]
		,input_nbr=[500]
		,tau_in = [0.03]
		,tau_out = [0.03]
		, alpha_p= [0.01]
		, alpha_n= [0.009]
		, beta_p= [1.5]
		, beta_n= [2.5]
		, tau_pre= [0.0168]
		, tau_post= [0.0337]
		, iterations=[1]
		, presentation_time = [0.20]
		, pause_time = [0.1]
		, dt = [0.005]
		, n_neurons = [10]
		, inhibition_time = [10]
		, tau_ref_in = [0.005]
		, tau_ref_out = [0.005]
		, inc_n = [0.01]
		, tau_n = [1]
		, synapse_layer_1=[0.005]
		, gain_in = [2]
		, gain_out = [2]
		, seed =[100]
    )
	param_values = [v for v in parameters.values()]

	now = time.strftime("%Y%m%d-%H%M%S")
	folder = os.getcwd()+"/MNIST_VDSP_explorartion"+now
	os.mkdir(folder)

	for args.amp_neuron,args.input_nbr,args.tau_in,args.tau_out,args.alpha_p,args.alpha_n,args.beta_p,args.beta_n,args.tau_pre,args.tau_post,args.iterations,args.presentation_time,args.pause_time, args.dt,args.n_neurons,args.inhibition_time,args.tau_ref_in,args.tau_ref_out,args.inc_n,args.tau_n,args.synapse_layer_1,args.gain_in,args.gain_out,args.seed in product(*param_values):


		# args.pause_time = 0

		# args.filename = 'vprog-'+str(args.vprog)+'-g_max-'+str(args.g_max)+'-tau_in-'+str(args.tau_in)+'-tau_out-'+str(args.tau_out)+'-lr-'+str(args.lr)+'-presentation_time-'+str(args.presentation_time)
		args.filename = 'stdp-'+str(args.amp_neuron)+str(args.input_nbr)+str(args.tau_in)+str(args.tau_out)+str(args.alpha_p)+str(args.alpha_n)+str(args.beta_p)+str(args.beta_n)+str(args.tau_pre)+str(args.tau_post)

		timestr = time.strftime("%Y%m%d-%H%M%S")
		log_file_name = 'accuracy_log'+'.csv'
		pwd = os.getcwd()

		accuracy, accuracy_2,weights = evaluate_mnist_multiple_baseline(args)

		df = df.append({ 
						"amp_neuron":args.amp_neuron,
						
						 "input_nbr":args.input_nbr,
						 "tau_in":args.tau_in,
						 "tau_out": args.tau_out,
					 
						 "alpha_p": args.alpha_p,
						 "alpha_n": args.alpha_n,
						 "beta_p":args.beta_p,
						 "beta_n": args.beta_n,
						 "tau_pre": args.tau_pre,
						 "tau_post": args.tau_post,
						 
						 "iterations":args.iterations,
		                 "presentation_time":args.presentation_time,
		                 "pause_time":args.pause_time,
		                 "dt":args.dt,
		                 "n_neurons":args.n_neurons,
		                 "seed":args.seed,
		                 "inhibition_time":args.inhibition_time,
		                 "tau_ref_in":args.tau_ref_in,
		                 "tau_ref_out":args.tau_ref_out,
		                 "inc_n":args.inc_n,
		                 "tau_n":args.tau_n,
		                 "synapse_layer_1":args.synapse_layer_1,
		                 
		                 "gain_in":args.gain_in,
		                 "bias_out":args.bias_out,

		                 "accuracy":accuracy,
		                 "accuracy_2":accuracy_2
		                 },ignore_index=True)
		

		plot = True
		if plot : 	
			print('accuracy', accuracy)
			print(args.filename)
			# weights = weights[-1]#Taking only the last weight for plotting

			columns = int(args.n_neurons/5)
			rows = int(args.n_neurons/columns)

			fig, axes = plt.subplots(int(args.n_neurons/columns), int(columns), figsize=(columns*5,rows*5))

			for i in range(0,(args.n_neurons)):
				axes[int(i/columns)][int(i%columns)].matshow(np.reshape(weights[i],(28,28)),interpolation='nearest')#, vmax=1, vmin=0)
				axes[int(i/columns)][int(i%columns)].get_xaxis().set_visible(False)
				axes[int(i/columns)][int(i%columns)].get_yaxis().set_visible(False)
			plt.tight_layout()    
			plt.axis('off')

	   
			# fig, axes = plt.subplots(1,1, figsize=(3,3))
			# fig = plt.figure()
			# ax1 = fig.add_subplot()
			# cax = ax1.matshow(np.reshape(weights[0],(28,28)),interpolation='nearest', vmax=1, vmin=0)
			# fig.colorbar(cax)
			# plt.tight_layout() 

			if args.log_file_path is None:
				log_dir = pwd+'/log_dir/'
			else : 
				log_dir = args.log_file_path
			df.to_csv(log_dir+log_file_name, index=False)   

			fig.savefig(log_dir+args.filename+'weights.png')
			plt.close()

			plt.clf()
			plt.hist(weights.flatten())

			plt.tight_layout()    
			plt.savefig(log_dir+args.filename+'histogram.png')


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