
import nengo
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from nengo.dists import Choice
from datetime import datetime
import pickle
from nengo.utils.matplotlib import rasterplot
import time
from InputData import PresentInputWithPause
from nengo.neurons import LIFRate
from nengo.params import Parameter, NumberParam, FrozenObject
from nengo.dists import Choice, Distribution, get_samples, Uniform
from nengo.utils.numpy import clip, is_array_like
from utilis import *
from args_mnist import args as my_args
import itertools
import random
import logging
import random 

def evaluate_fmnist_multiple_tio2(args):

    #############################
    # load the data
    #############################
    input_nbr = args.input_nbr
    input_nbr = args.input_nbr

    probe_sample_rate = (input_nbr/10)/1000 #Probe sample rate. Proportional to input_nbr to scale down sampling rate of simulations 


    x = args.digit
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = np.load('fmnist_norm.npz', allow_pickle=True)
    image_train_filtered = data['image_train_filtered']/255
    label_train_filtered = data['label_train_filtered']
    image_test_filtered = data['image_test_filtered']/255
    label_test_filtered = data['label_test_filtered']

    image_train_filtered = np.tile(image_train_filtered,(args.iterations,1,1))
    label_train_filtered = np.tile(label_train_filtered,(args.iterations))

    #Simulation Parameters 
    #Presentation time
    presentation_time = args.presentation_time #0.20
    #Pause time
    # pause_time = args.pause_time + 0.0001
    pause_time = args.pause_time
    #Iterations
    iterations=args.iterations
    #Input layer parameters
    n_in = args.n_in
    # g_max = 1/784 #Maximum output contribution
    amp_neuron = args.amp_neuron
    n_neurons = args.n_neurons # Layer 1 neurons
    # inhib_factor = args.inhib_factor #Multiplication factor for lateral inhibition


    input_neurons_args = {
            "n_neurons":n_in,
            "dimensions":1,
            "label":"Input layer",
            "encoders":nengo.dists.Choice([[1]]),
            # "max_rates":nengo.dists.Uniform(22,22),
            # "intercepts":nengo.dists.Uniform(0,0),
            "gain":nengo.dists.Choice([args.gain_in]),
            "bias":nengo.dists.Choice([args.bias_in]),
            # "noise":nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(args.noise_input, (args.noise_input/2)+0.00001), seed=1), 

            "neuron_type":MyLIF_in(tau_rc=args.tau_in,min_voltage=-1.8, amplitude=args.amp_neuron, tau_ref=args.tau_ref_in)
            # "neuron_type":nengo.neurons.SpikingRectifiedLinear()#SpikingRelu neuron. 
    }

    #Layer 1 parameters
    layer_1_neurons_args = {
            "n_neurons":n_neurons,
            "dimensions":1,
            "label":"Layer 1",
            "encoders":nengo.dists.Choice([[1]]),
            "gain":nengo.dists.Choice([args.gain_out]),
            "bias":nengo.dists.Choice([args.bias_out]),
            # "intercepts":nengo.dists.Choice([0]),
            # "max_rates":nengo.dists.Choice([args.rate_out,args.rate_out]),
            # "noise":nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, 0.5), seed=1), 
            # "neuron_type":nengo.neurons.LIF(tau_rc=args.tau_out, min_voltage=0)
            # "neuron_type":MyLIF_out(tau_rc=args.tau_out, min_voltage=-1)
            "neuron_type":STDPLIF(tau_rc=args.tau_out, min_voltage=-1, spiking_threshold=args.thr_out, inhibition_time=args.inhibition_time,tau_ref=args.tau_ref_out,inc_n=args.inc_n,tau_n=args.tau_n)
    }

    #Learning rule parameters
    learning_args = {
            "lr": args.lr,
            "winit_min":0,
            "winit_max":args.winit_max,
            "vprog":args.vprog, 
            "vthp":0.5,
            "vthn":0.5,
            "gmax":0.0008,
            "gmin":0.00008,
            "vprog_increment":args.vprog_increment,
            "voltage_clip_max":args.voltage_clip_max,
            "voltage_clip_min":args.voltage_clip_min,
            "Vapp_multiplier":args.Vapp_multiplier,
            "sample_distance": int((presentation_time+pause_time)*200*10), #Store weight after 10 images
    }

    # argument_string = "presentation_time: "+ str(presentation_time)+ "\n pause_time: "+ str(pause_time)+ "\n input_neurons_args: " + str(input_neurons_args)+ " \n layer_1_neuron_args: " + str(layer_1_neurons_args)+"\n Lateral Inhibition parameters: " + str(lateral_inhib_args) + "\n learning parameters: " + str(learning_args)+ "\n g_max: "+ str(g_max) 

    images = image_train_filtered
    labels = label_train_filtered
    np.random.seed(args.seed)
    random.seed(args.seed) 

    model = nengo.Network("My network", seed = args.seed)
    #############################
    # Model construction
    #############################
    with model:
        # picture = nengo.Node(PresentInputWithPause(images, presentation_time, pause_time,0))
        picture = nengo.Node(nengo.processes.PresentInput(images, presentation_time=presentation_time))
        true_label = nengo.Node(nengo.processes.PresentInput(labels, presentation_time=presentation_time))
        # true_label = nengo.Node(PresentInputWithPause(labels, presentation_time, pause_time,-1))
        # input layer  
        input_layer = nengo.Ensemble(**input_neurons_args)
        input_conn = nengo.Connection(picture,input_layer.neurons,synapse=None)
        #first layer
        layer1 = nengo.Ensemble(**layer_1_neurons_args)
        #Weights between input layer and layer 1
        w = nengo.Node(CustomRule_post_v2_tio2(**learning_args), size_in=n_in, size_out=n_neurons)
        nengo.Connection(input_layer.neurons, w, synapse=None)
        nengo.Connection(w, layer1.neurons, synapse=args.synapse_layer_1)
        weights = w.output.history
        
    # with nengo_ocl.Simulator(model) as sim :   
    with nengo.Simulator(model, dt=args.dt, optimize=True) as sim:

        
        w.output.set_signal_vmem(sim.signals[sim.model.sig[input_layer.neurons]["voltage"]])
        w.output.set_signal_out(sim.signals[sim.model.sig[layer1.neurons]["out"]])
        sim.run((presentation_time+pause_time) * labels.shape[0])

    last_weight = weights[-1]

    sim.close()

    pause_time = 0

    #Neuron class assingment

    images = image_train_filtered
    labels = label_train_filtered


    model = nengo.Network("My network", seed = args.seed)

    with model:

        # picture = nengo.Node(PresentInputWithPause(images, presentation_time, pause_time,0))
        picture = nengo.Node(nengo.processes.PresentInput(images, presentation_time=presentation_time))
        true_label = nengo.Node(nengo.processes.PresentInput(labels, presentation_time=presentation_time))
        # true_label = nengo.Node(PresentInputWithPause(labels, presentation_time, pause_time,-1))
        # input layer  
        input_layer = nengo.Ensemble(**input_neurons_args)
        input_conn = nengo.Connection(picture,input_layer.neurons,synapse=None)
        #first layer
        layer1 = nengo.Ensemble(**layer_1_neurons_args)
        nengo.Connection(input_layer.neurons, layer1.neurons,transform=last_weight,synapse=args.synapse_layer_1)
        #Probes
        p_true_label = nengo.Probe(true_label)
        p_layer_1 = nengo.Probe(layer1.neurons)

    # with nengo_ocl.Simulator(model) as sim :   
    with nengo.Simulator(model, dt=args.dt, optimize=True) as sim:
        
        sim.run((presentation_time+pause_time) * labels.shape[0])
    
    t_data = sim.trange()
    labels = sim.data[p_true_label][:,0]
    output_spikes = sim.data[p_layer_1]
    neuron_class = np.zeros((n_neurons, 1))
    n_classes = 10
    for j in range(n_neurons):
        spike_times_neuron_j = t_data[np.where(output_spikes[:,j] > 0)]
        max_spike_times = 0 
        for i in range(n_classes):
            class_presentation_times_i = t_data[np.where(labels == i)]
            #Normalized number of spikes wrt class presentation time
            num_spikes = len(np.intersect1d(spike_times_neuron_j,class_presentation_times_i))/(len(class_presentation_times_i)+1)
            if(num_spikes>max_spike_times):
                neuron_class[j] = i
                max_spike_times = num_spikes
    spikes_layer1_probe_train = sim.data[p_layer_1]



    #Testing

    images = image_test_filtered
    labels = label_test_filtered



    input_nbr = 10000
    
    model = nengo.Network(label="My network",)

    with model:

        # picture = nengo.Node(PresentInputWithPause(images, presentation_time, pause_time,0))
        picture = nengo.Node(nengo.processes.PresentInput(images, presentation_time=presentation_time))
        true_label = nengo.Node(nengo.processes.PresentInput(labels, presentation_time=presentation_time))
        # true_label = nengo.Node(PresentInputWithPause(labels, presentation_time, pause_time,-1))
        input_layer = nengo.Ensemble(**input_neurons_args)
        input_conn = nengo.Connection(picture,input_layer.neurons,synapse=None)
        #first layer
        layer1 = nengo.Ensemble(**layer_1_neurons_args)
        nengo.Connection(input_layer.neurons, layer1.neurons,transform=last_weight,synapse=args.synapse_layer_1)
        p_true_label = nengo.Probe(true_label)
        p_layer_1 = nengo.Probe(layer1.neurons)


    step_time = (presentation_time + pause_time) 

    with nengo.Simulator(model,dt=args.dt) as sim:
           
        sim.run(presentation_time * label_test_filtered.shape[0])

    accuracy_2 = evaluation_v2(10,n_neurons,int(((presentation_time * label_test_filtered.shape[0]) / sim.dt) / input_nbr),spikes_layer1_probe_train,label_train_filtered,sim.data[p_layer_1],label_test_filtered,sim.dt)


    labels = sim.data[p_true_label][:,0]
    t_data = sim.trange()
    output_spikes = sim.data[p_layer_1]
    n_classes = 10
    predicted_labels = []  
    true_labels = []
    correct_classified = 0
    wrong_classified = 0

    class_spikes = np.ones((10,1))

    for num in range(input_nbr):
        #np.sum(sim.data[my_spike_probe] > 0, axis=0)

        output_spikes_num = output_spikes[num*int((presentation_time + pause_time) /args.dt):(num+1)*int((presentation_time + pause_time) /args.dt),:] # 0.350/0.005
        num_spikes = np.sum(output_spikes_num > 0, axis=0)

        for i in range(n_classes):
            sum_temp = 0
            count_temp = 0
            for j in range(n_neurons):
                if((neuron_class[j]) == i) : 
                    sum_temp += num_spikes[j]
                    count_temp +=1
        
            if(count_temp==0):
                class_spikes[i] = 0
            else:
                class_spikes[i] = sum_temp
                # class_spikes[i] = sum_temp/count_temp

        # print(class_spikes)
        k = np.argmax(num_spikes)
        # predicted_labels.append(neuron_class[k])
        class_pred = np.argmax(class_spikes)
        predicted_labels.append(class_pred)

        true_class = labels[(num*int((presentation_time + pause_time) /args.dt))]
        # print(true_class)
        # print(class_pred)

        # if(neuron_class[k] == true_class):
        #     correct_classified+=1
        # else:
        #     wrong_classified+=1
        if(class_pred == true_class):
            correct_classified+=1
        else:
            wrong_classified+=1

        
    accuracy = correct_classified/ (correct_classified+wrong_classified)*100
    print("Accuracy: ", accuracy)
    sim.close()

    del weights, sim.data, labels, class_pred, spikes_layer1_probe_train

    return accuracy, accuracy_2, last_weight


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



    # params = nni.get_next_parameter()

    # args.g_max = params['g_max']
    # args.tau_in = params['tau_in']
    # args.tau_out = params['tau_out']
    # args.lr = params['lr']
    # args.presentation_time = params['presentation_time']
    # args.rate_out = params['rate_out']



    accuracy, weights = evaluate_mnist_multiple(args)
    print('accuracy:', accuracy)

    # now = time.strftime("%Y%m%d-%H%M%S")
    # folder = os.getcwd()+"/MNIST_VDSP"+now
    # os.mkdir(folder)


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

    # plt.savefig(folder+'/raster'+'.png')


    # for tstep in np.arange(0, len(weights), 1):
    #     tstep = int(tstep)
    #     # tstep = len(weightds) - tstep -1


    #     print(tstep)

    #     columns = int(args.n_neurons/5)
    #     fig, axes = plt.subplots(int(args.n_neurons/columns), int(columns), figsize=(20,25))

    #     for i in range(0,(args.n_neurons)):

    #         axes[int(i/columns)][int(i%columns)].matshow(np.reshape(weights[tstep][i],(28,28)),interpolation='nearest', vmax=1, vmin=0)


    #     plt.tight_layout()    
    #     fig.savefig(folder+'/weights'+str(tstep)+'.png')
    #     plt.close('all')

    # gen_video(folder, "weights")



    logger.info('All done.')