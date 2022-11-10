import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import nengo
import numpy as np
from numpy import random
#from src.Models.Neuron.STDPLIF import STDPLIF
#from DataLog import DataLog
from InputData import PresentInputWithPause
# from Heatmap import AllHeatMapSave,HeatMapSave
from nengo.dists import Choice
from datetime import datetime
from nengo_extras.data import load_mnist
from utilis import *
import pickle
import tensorflow as tf

#############################
# load the data
#############################

img_rows, img_cols = 28, 28
input_nbr = 60000
iterations = 1
probe_sample_rate = (input_nbr/10)/1000 #Probe sample rate. Proportional to input_nbr to scale down sampling rate of simulations 

dt = 0.005
learning_rate = 0.1
# learning_rate=0.125
vprog = -0.75
n_neurons = 100




Dataset = "Mnist"
# (image_train, label_train), (image_test, label_test) = load_mnist()
(image_train, label_train), (image_test, label_test) = (tf.keras.datasets.mnist.load_data())


#select the 0s and 1s as the two classes from MNIST data
image_train_filtered = []
label_train_filtered = []

for i in range(0,input_nbr):
#  if (label_train[i] == 1 or label_train[i] == 0):
        image_train_filtered.append(image_train[i])
        label_train_filtered.append(label_train[i])

print("actual input",len(label_train_filtered))
print(np.bincount(label_train_filtered))

image_train_filtered = np.array(image_train_filtered)
label_train_filtered = np.array(label_train_filtered)

#############################

model = nengo.Network(label="My network",)

#############################
# Helpfull methodes
#############################

def sparsity_measure(vector):  # Gini index
    # Max sparsity = 1 (single 1 in the vector)
    v = np.sort(np.abs(vector))
    n = v.shape[0]
    k = np.arange(n) + 1
    l1norm = np.sum(v)
    summation = np.sum((v / l1norm) * ((n - k + 0.5) / n))
    return 1 - 2 * summation

#############################
# Model construction
#############################

presentation_time = 0.35 #0.35
pause_time = 0 #0.15
#input layer
n_in = 784

# Learning params


learning_args = {
            "lr": learning_rate,
            "winit_min":0,
            "winit_max":1,
            "vprog":vprog,
            "vthp":0.16,
            "vthn":0.15,
            "sample_distance": int((presentation_time+pause_time)*200*100),#Save weights after 100 images
    }


#    log = DataLog()
with model:
    # input layer 
      # picture = nengo.Node(PresentInputWithPause(images, presentation_time, pause_time,0))
    picture = nengo.Node(nengo.processes.PresentInput(image_train_filtered, presentation_time=presentation_time))
    true_label = nengo.Node(nengo.processes.PresentInput(label_train_filtered, presentation_time=presentation_time))
        # true_label = nengo.Node(PresentInputWithPause(labels, presentation_time, pause_time,-1))
    input_layer = nengo.Ensemble(
        n_in,
        1,
        label="Input",
        neuron_type=MyLIF_in(tau_rc=0.06,min_voltage=-1,amplitude=1/210),#nengo.neurons.PoissonSpiking(nengo.LIFRate(amplitude=0.2)),#nengo.LIF(amplitude=0.2),# nengo.neurons.PoissonSpiking(nengo.LIFRate(amplitude=0.2))
        gain=nengo.dists.Choice([2]),
        encoders=nengo.dists.Choice([[1]]),
        bias=nengo.dists.Choice([0]))

    input_conn = nengo.Connection(picture,input_layer.neurons)

    # define first layer
    layer1 = nengo.Ensemble(
         n_neurons,
         1,
         label="layer1",
         neuron_type=STDPLIF(tau_rc=0.06, min_voltage=-1, spiking_threshold = 1),
         encoders=nengo.dists.Choice([[1]]),

         # max_rates=nengo.dists.Choice([20]),
         # intercepts=nengo.dists.Choice([0]),
         gain=nengo.dists.Choice([2]),
         bias=nengo.dists.Choice([0])
         )

    # init_weights = np.random.uniform(0, 1, (n_neurons, n_in))
    
    w = nengo.Node(CustomRule_post_v2(**learning_args), size_in=784, size_out=n_neurons)  
    nengo.Connection(input_layer.neurons, w)
    nengo.Connection(w, layer1.neurons)

    # conn1 = nengo.Connection(input_layer.neurons,layer1.neurons,learning_rule_type=VLR(learning_rate=learning_rate,vprog=-0.6),transform=init_weights)

    p_true_label = nengo.Probe(true_label, sample_every=probe_sample_rate)
    p_layer_1 = nengo.Probe(layer1.neurons, sample_every=probe_sample_rate)
    
    weights = w.output.history


    # weights_probe = nengo.Probe(conn1,"weights",sample_every=probe_sample_rate)
    #if(not full_log):
    #    nengo.Node(log)

    #############################

step_time = (presentation_time + pause_time) 
Args = {"Dataset":Dataset,"Labels":label_train_filtered,"step_time":step_time,"input_nbr":input_nbr}

with nengo.Simulator(model,dt=dt) as sim:
    
    #if(not full_log):
    #    log.set(sim,Args,False,False)

    w.output.set_signal_vmem(sim.signals[sim.model.sig[input_layer.neurons]["voltage"]])
    w.output.set_signal_out(sim.signals[sim.model.sig[layer1.neurons]["out"]])
    
    sim.run(iterations*step_time * label_train_filtered.shape[0])


# weights = sim.data[weights_probe][-1]
weights = weights[-1]
#if(not full_log):
#    log.closeLog()

#print("Prune : ",np.round(100 - (((n_in * n_neurons) - np.sum(np.array(w.output.pruned)))* 100 / (n_in * n_neurons)),2)," %")
now = str(datetime.now().time())
folder = "My_Sim_"+"learning_rate="+str(learning_rate)+"_"+now

if not os.path.exists(folder):
    os.makedirs(folder)
    

i = 0
#np.putmask(weights,w.output.pruned == 1,np.NAN)

#save the pruned model
# pickle.dump(sim.data[connection_layer1_probe][-1], open( "mnist_params_STDP", "wb" ))
pickle.dump(weights, open( "mnist_params_STDP", "wb" ))
for n in weights:
    plt.matshow(np.reshape(n,(28,28)),interpolation='none',vmax=1, vmin=0)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(folder+"/"+str(i)+".svg")
    plt.cla()
    i = i + 1


t_data = sim.trange(sample_every=probe_sample_rate)
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
            
print("Neuron class: \n", neuron_class)


'''
Testing
'''

img_rows, img_cols = 28, 28
input_nbr = int(input_nbr/6)

Dataset = "Mnist"
# (image_train, label_train), (image_test, label_test) = load_mnist()
(image_train, label_train), (image_test, label_test) = (tf.keras.datasets.mnist.load_data())


#select the 0s and 1s as the two classes from MNIST data
image_test_filtered = []
label_test_filtered = []

for i in range(0,input_nbr):
#  if (label_train[i] == 1 or label_train[i] == 0):
    image_test_filtered.append(image_test[i])
    label_test_filtered.append(label_test[i])

print("actual input",len(label_test_filtered))
print(np.bincount(label_test_filtered))

image_test_filtered = np.array(image_test_filtered)
label_test_filtered = np.array(label_test_filtered)

#############################

model = nengo.Network(label="My network",)



# Learning params

with model:
    # input layer 
      # picture = nengo.Node(PresentInputWithPause(images, presentation_time, pause_time,0))
    picture = nengo.Node(nengo.processes.PresentInput(image_test_filtered, presentation_time=presentation_time))
    true_label = nengo.Node(nengo.processes.PresentInput(label_test_filtered, presentation_time=presentation_time))
        # true_label = nengo.Node(PresentInputWithPause(labels, presentation_time, pause_time,-1))
    input_layer = nengo.Ensemble(
        n_in,
        1,
        label="Input",
        neuron_type=MyLIF_in(tau_rc=0.06,min_voltage=-1,amplitude=1/210),#nengo.neurons.PoissonSpiking(nengo.LIFRate(amplitude=0.2)),#nengo.LIF(amplitude=0.2),# nengo.neurons.PoissonSpiking(nengo.LIFRate(amplitude=0.2))
        gain=nengo.dists.Choice([2]),
        encoders=nengo.dists.Choice([[1]]),
        bias=nengo.dists.Choice([0]))

    input_conn = nengo.Connection(picture,input_layer.neurons,)

    # weights randomly initiated 
    #layer1_weights = np.round(random.random((n_neurons, 784)),5)
    # define first layer
    layer1 = nengo.Ensemble(
         n_neurons,
         1,
         label="layer1",
         neuron_type=STDPLIF(tau_rc=0.1, min_voltage=-1),
         gain=nengo.dists.Choice([2]),
        encoders=nengo.dists.Choice([[1]]),
        bias=nengo.dists.Choice([0]))

    # w = nengo.Node(CustomRule_post_v2(**learning_args), size_in=784, size_out=n_neurons)
    
    nengo.Connection(input_layer.neurons, layer1.neurons,transform=weights)
 
   
    p_true_label = nengo.Probe(true_label)
    p_layer_1 = nengo.Probe(layer1.neurons)
    #if(not full_log):
    #    nengo.Node(log)

    #############################

step_time = (presentation_time + pause_time) 

with nengo.Simulator(model,dt=dt) as sim:
       
    sim.run(step_time * label_test_filtered.shape[0])


labels = sim.data[p_true_label][:,0]
output_spikes = sim.data[p_layer_1]
n_classes = 10
# rate_data = nengo.synapses.Lowpass(0.1).filtfilt(sim.data[p_layer_1])
predicted_labels = []  
true_labels = []
correct_classified = 0
wrong_classified = 0


class_spikes = np.ones((10,1))

for num in range(input_nbr):
    #np.sum(sim.data[my_spike_probe] > 0, axis=0)

    output_spikes_num = output_spikes[num*int(presentation_time/dt):(num+1)*int(presentation_time/dt),:] # 0.350/0.005
    num_spikes = np.sum(output_spikes_num > 0, axis=0)

    for i in range(n_classes):
    	sum_temp = 0
    	count_temp = 0
    	for j in range(n_neurons):
    		if((neuron_class[j]) == i) : 
    			sum_temp += num_spikes[j]
    			count_temp +=1

    	class_spikes[i] = sum_temp
        # class_spikes[i] = sum_temp/count_temp



    # print(class_spikes)
    k = np.argmax(num_spikes)
    # predicted_labels.append(neuron_class[k])
    class_pred = np.argmax(class_spikes)
    predicted_labels.append(class_pred)

    true_class = labels[(num*int(presentation_time/dt))]

    if(class_pred == true_class):
        correct_classified+=1
    else:
        wrong_classified+=1
        
accuracy = correct_classified/ (correct_classified+wrong_classified)*100
print("Accuracy: ", accuracy)
#Ratio = Ratio + (alpha * (CRmaining / CTotal))
