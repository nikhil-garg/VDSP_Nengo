
import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from fmnist_vdsp_multiple_tio2 import *
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
						"amp_neuron":[],
						"vth":[],
						"input_nbr":[],
						"tau_in" :[],
						"tau_out":[],
                        "lr":[],
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
                        "winit_max":[],
                        "vprog_increment":[],
                        "voltage_clip_max":[],
                        "voltage_clip_min":[],
                        "Vapp_multiplier":[],
                        "gain_in":[],
                        "bias_in":[],
                        "noise_input":[],
                        "accuracy":[],
                        "accuracy_2":[]
                         })

	if args.log_file_path is None:
		log_dir = pwd+'/log_dir/'
	else : 
		log_dir = args.log_file_path
		df.to_csv(log_dir+'test.csv', index=False)

	parameters = dict(
		vprog = [0]
		, amp_neuron=[0.5]
		,input_nbr=[60000]
		,tau_in = [0.03]
		,tau_out = [0.03]
		, lr = [1]
		, iterations=[1]
		, presentation_time = [0.35]
		, pause_time = [0]
		, dt = [0.005]
		, n_neurons = [200]
		, inhibition_time = [10]
		, tau_ref_in = [0.01]
		, tau_ref_out = [0.005]
		, inc_n = [0.01]
		, tau_n = [1]
		, synapse_layer_1=[0.005]
		, winit_max = [1]
		, vprog_increment = [0]
		, voltage_clip_max=[1.8]
		, voltage_clip_min = [-1.5]
		, Vapp_multiplier = [1]
		, gain_in = [3.5]
		, bias_in = [0.85]
		, noise_input = [0]
		, seed =[100]
    )
	param_values = [v for v in parameters.values()]

	now = time.strftime("%Y%m%d-%H%M%S")
	folder = os.getcwd()+"/MNIST_VDSP_explorartion"+now
	os.mkdir(folder)

	for args.vprog,args.amp_neuron,args.input_nbr,args.tau_in,args.tau_out,args.lr,args.iterations,args.presentation_time,args.pause_time, args.dt,args.n_neurons,args.inhibition_time,args.tau_ref_in,args.tau_ref_out,args.inc_n,args.tau_n,args.synapse_layer_1,args.winit_max,args.vprog_increment,args.voltage_clip_max,args.voltage_clip_min,args.Vapp_multiplier,args.gain_in,args.bias_in,args.noise_input,args.seed in product(*param_values):


		# args.pause_time = 0

		# args.filename = 'vprog-'+str(args.vprog)+'-g_max-'+str(args.g_max)+'-tau_in-'+str(args.tau_in)+'-tau_out-'+str(args.tau_out)+'-lr-'+str(args.lr)+'-presentation_time-'+str(args.presentation_time)
		args.filename = 'vprog-'+str(args.vprog)+'amp_neuron'+str(args.amp_neuron)+'-tau_in-'+str(args.tau_in)+'-tau_out-'+str(args.tau_out)+'-lr-'+str(args.lr)+'-presentation_time-'+str(args.presentation_time)+'pause_time'+str(args.pause_time) + 'dt-'+str(args.dt)+'ref-'+str(args.tau_ref_in)+str(args.tau_ref_out)+'gain-'+str(args.gain_in)+'bias_in'+str(args.bias_in)+'adaptation'+str(args.inc_n)+str(args.tau_n)+'noise'+str(args.noise_input)+'Vapp_multiplier-'+str(args.Vapp_multiplier)+'winit_max'+str(args.winit_max)+str(args.voltage_clip_max)+str(args.voltage_clip_min)+str(args.n_neurons)+str(args.seed)

		timestr = time.strftime("%Y%m%d-%H%M%S")
		log_file_name = 'accuracy_log'+'.csv'
		pwd = os.getcwd()

		accuracy, accuracy_2,weights = evaluate_fmnist_multiple_tio2(args)

		df = df.append({ "vprog":args.vprog,
						"amp_neuron":args.amp_neuron,
						 "vth":args.vthp,
						 "input_nbr":args.input_nbr,
						 "tau_in":args.tau_in,
						 "tau_out": args.tau_out,
						 "lr": args.lr,
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
		                 "winit_max":args.winit_max,
		                 "vprog_increment":args.vprog_increment,
		                 "voltage_clip_max":args.voltage_clip_max,
		                 "voltage_clip_min":args.voltage_clip_min,
		                 "Vapp_multiplier":args.Vapp_multiplier,
		                 "gain_in":args.gain_in,
		                 "bias_in":args.bias_in,
		                 "noise_input":args.noise_input,
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
				axes[int(i/columns)][int(i%columns)].matshow(np.reshape(weights[i],(28,28)),interpolation='nearest', vmax=1, vmin=0, cmap='hot')

			plt.tight_layout()    

	   
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