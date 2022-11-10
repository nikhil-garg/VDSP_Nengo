
import nengo
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import os
from nengo.dists import Choice
from datetime import datetime
from nengo_extras.data import load_mnist
import pickle
from nengo.utils.matplotlib import rasterplot

import time

from InputData import PresentInputWithPause

from nengo_extras.graphviz import net_diagram

from nengo.neurons import LIFRate

from nengo.params import Parameter, NumberParam, FrozenObject
from nengo.dists import Choice, Distribution, get_samples, Uniform

from nengo.utils.numpy import clip, is_array_like
from utilis import *


from args_mnist import args as my_args
import itertools
import random
import logging

# import nengo_spinnaker
import nengo_ocl

def evaluate_mnist_single(args):

    #############################
    # load the data
    #############################
    input_nbr = args.input_nbr

    (image_train, label_train), (image_test, label_test) = (tf.keras.datasets.fashion_mnist.load_data())

    probe_sample_rate = (input_nbr/10)/1000 #Probe sample rate. Proportional to input_nbr to scale down sampling rate of simulations 
    # probe_sample_rate = 1000
    image_train_filtered = []
    label_train_filtered = []

    x = args.digit

    for i in range(0,input_nbr):
      if label_train[i] == x:
            image_train_filtered.append(image_train[i])
            label_train_filtered.append(label_train[i])

    image_train_filtered = np.array(image_train_filtered)
    label_train_filtered = np.array(label_train_filtered)


    #Simulation Parameters 
    #Presentation time
    presentation_time = args.presentation_time #0.20
    #Pause time
    pause_time = args.pause_time
    #Iterations
    iterations=args.iterations
    #Input layer parameters
    n_in = args.n_in
    # g_max = 1/784 #Maximum output contribution
    g_max = args.g_max
    n_neurons = args.n_neurons # Layer 1 neurons
    inhib_factor = -0*100 #Multiplication factor for lateral inhibition

    n_neurons = 1
    input_neurons_args = {
            "n_neurons":n_in,
            "dimensions":1,
            "label":"Input layer",
            "encoders":nengo.dists.Uniform(1,1),
            "gain":nengo.dists.Uniform(2,2),
            "bias":nengo.dists.Uniform(0,0),
            "neuron_type":MyLIF_in(tau_rc=args.tau_in,min_voltage=-1, amplitude=args.g_max)
            # "neuron_type":nengo.neurons.LIF(tau_rc=args.tau_in,min_voltage=0)#SpikingRelu neuron. 
    }

    #Layer 1 parameters
    layer_1_neurons_args = {
            "n_neurons":n_neurons,
            "dimensions":1,
            "label":"Layer 1",
            "encoders":nengo.dists.Uniform(1,1),
            # "gain":nengo.dists.Uniform(2,2),
            # "bias":nengo.dists.Uniform(0,0),
            "intercepts":nengo.dists.Choice([0]),
            "max_rates":nengo.dists.Choice([20,20]),
            "noise":nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, 1), seed=1), 
            # "neuron_type":nengo.neurons.LIF(tau_rc=args.tau_out, min_voltage=0)
            # "neuron_type":MyLIF_out(tau_rc=args.tau_out, min_voltage=-1)
            "neuron_type":STDPLIF(tau_rc=args.tau_out, min_voltage=-1),
    }


    # "noise":nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, 20), seed=1),     

    #Lateral Inhibition parameters
    lateral_inhib_args = {
            "transform": inhib_factor* (np.full((n_neurons, n_neurons), 1) - np.eye(n_neurons)),
            "synapse":0.01,
            "label":"Lateral Inhibition"
    }

    #Learning rule parameters
    learning_args = {
            "lr": args.lr,
            "winit_min":0,
            "winit_max":0.1,
    #         "tpw":50,
    #         "prev_flag":True,
            "sample_distance": int((presentation_time+pause_time)*200),
    }

    argument_string = "presentation_time: "+ str(presentation_time)+ "\n pause_time: "+ str(pause_time)+ "\n input_neurons_args: " + str(input_neurons_args)+ " \n layer_1_neuron_args: " + str(layer_1_neurons_args)+"\n Lateral Inhibition parameters: " + str(lateral_inhib_args) + "\n learning parameters: " + str(learning_args)+ "\n g_max: "+ str(g_max) 

    images = image_train_filtered
    labels = label_train_filtered


    model = nengo.Network("My network")
    #############################
    # Model construction
    #############################
    with model:

        picture = nengo.Node(nengo.processes.PresentInput(images, presentation_time))
        true_label = nengo.Node(nengo.processes.PresentInput(labels, presentation_time))
        # picture = nengo.Node(PresentInputWithPause(images, presentation_time, pause_time,0))
        # true_label = nengo.Node(PresentInputWithPause(labels, presentation_time, pause_time,-1))

        # input layer  
        input_layer = nengo.Ensemble(**input_neurons_args)
        input_conn = nengo.Connection(picture,input_layer.neurons,synapse=None)

        #first layer
        layer1 = nengo.Ensemble(**layer_1_neurons_args)

        #Weights between input layer and layer 1
        w = nengo.Node(CustomRule_post_v2(**learning_args), size_in=n_in, size_out=n_neurons)
        nengo.Connection(input_layer.neurons, w, synapse=None)
        nengo.Connection(w, layer1.neurons, synapse=None)

        #Lateral inhibition
        # inhib = nengo.Connection(layer1.neurons,layer1.neurons,**lateral_inhib_args) 

        #Probes
        # p_true_label = nengo.Probe(true_label, sample_every=probe_sample_rate)
        p_input_layer = nengo.Probe(input_layer.neurons, sample_every=probe_sample_rate)
        p_layer_1 = nengo.Probe(layer1.neurons, sample_every=probe_sample_rate)
        weights = w.output.history

        


    with nengo.Simulator(model, dt=0.005) as sim :   
    # with nengo_spinnaker.Simulator(model) as sim:

        
        w.output.set_signal_vmem(sim.signals[sim.model.sig[input_layer.neurons]["voltage"]])
        w.output.set_signal_out(sim.signals[sim.model.sig[layer1.neurons]["out"]])
        
        
        sim.run((presentation_time+pause_time) * labels.shape[0]*iterations)

    

    #save the model
    # now = time.strftime("%Y%m%d-%H%M%S")
    # folder = os.getcwd()+"/MNIST_VDSP"+now
    # os.mkdir(folder)
    last_weight = weights[-1]

    # pickle.dump(weights, open( folder+"/trained_weights", "wb" ))
    # pickle.dump(argument_string, open( folder+"/arguments", "wb" ))

    sim.close()

    return weights


    # for tstep in np.arange(0, len(weights), 1):
    #     tstep = int(tstep)
    #     print(tstep)
    #     fig, axes = plt.subplots(1,1, figsize=(3,3))

    #     for i in range(0,(n_neurons)):
            
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot()
    #         cax = ax1.matshow(np.reshape(weights[tstep][i],(28,28)),interpolation='nearest', vmax=1, vmin=0)
    #         fig.colorbar(cax)

    #     plt.tight_layout()    
    #     fig.savefig(folder+'/weights'+str(tstep)+'.png')
    #     plt.close('all')

    # gen_video(folder, "weights")



    # for tstep in np.arange(0, len(weights), 1):
    #     tstep = int(tstep)
    #     print(tstep)
    #     fig, axes = plt.subplots(1,1, figsize=(3,3))

    #     for i in range(0,(n_neurons)):
            
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot()
    #         cax = ax1.hist(weights[tstep][i])
    #         ax1.set_xlim(0,1)
    #         ax1.set_ylim(0,350)

    #     plt.tight_layout()    
    #     fig.savefig(folder+'/histogram'+str(tstep)+'.png')
    #     plt.close('all')

    # gen_video(folder, "histogram")



if __name__ == '__main__':
	logger = logging.getLogger(__name__)

	args = my_args()
	print(args.__dict__)
	logging.basicConfig(level=logging.DEBUG)
	# Fix the seed of all random number generator
	seed = 500
	random.seed(seed)
	np.random.seed(seed)
	weights = evaluate_mnist_single(args)

	now = time.strftime("%Y%m%d-%H%M%S")
	folder = os.getcwd()+"/MNIST_VDSP_single"+now
	os.mkdir(folder)



	for tstep in np.arange(0, len(weights), 1):
		tstep = int(tstep)
		print(tstep)
		fig, axes = plt.subplots(1,1, figsize=(3,3))
		n_neurons = 1
		for i in range(0,(n_neurons)):
		    
		    fig = plt.figure()
		    ax1 = fig.add_subplot()
		    cax = ax1.matshow(np.reshape(weights[tstep][i],(28,28)),interpolation='nearest', vmax=1, vmin=0)
		    fig.colorbar(cax)

		plt.tight_layout()    
		fig.savefig(folder+'/weights'+str(tstep)+'.png')
		plt.close('all')

	gen_video(folder, "weights")



	for tstep in np.arange(0, len(weights), 1):
	    tstep = int(tstep)
	    print(tstep)
	    fig, axes = plt.subplots(1,1, figsize=(3,3))

	    for i in range(0,(n_neurons)):
	        
	        fig = plt.figure()
	        ax1 = fig.add_subplot()
	        cax = ax1.hist(weights[tstep][i])
	        ax1.set_xlim(0,1)
	        ax1.set_ylim(0,350)

	    plt.tight_layout()    
	    fig.savefig(folder+'/histogram'+str(tstep)+'.png')
	    plt.close('all')

	gen_video(folder, "histogram")

	logger.info('All done.')
