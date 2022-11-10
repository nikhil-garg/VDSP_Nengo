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
import pandas as pd

presentation_time = 0.35
pause_time = 0

img_rows, img_cols = 28, 28
input_nbr = 10000
n_in = 784
n_neurons = 30

weights =  pd.read_pickle("mnist_params_STDP")

neuron_class = np.array((7,
2,
6,
1,
6,
4,
4,
5,
1,
3,
6,
7,
5,
9,
6,
8,
4,
9,
1,
0,
2,
1,
5,
2,
7,
0,
9,
3,
7,
3))

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
        neuron_type=MyLIF_in(tau_rc=0.3,min_voltage=-2,amplitude=0.3),#nengo.neurons.PoissonSpiking(nengo.LIFRate(amplitude=0.2)),#nengo.LIF(amplitude=0.2),# nengo.neurons.PoissonSpiking(nengo.LIFRate(amplitude=0.2))
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
         neuron_type=STDPLIF(tau_rc=0.3, min_voltage=-1),
         intercepts=nengo.dists.Choice([0]),
         max_rates=nengo.dists.Choice([20,20]),
         encoders=nengo.dists.Choice([[1]]))

    # w = nengo.Node(CustomRule_post_v2(**learning_args), size_in=784, size_out=n_neurons)
    
    nengo.Connection(input_layer.neurons, layer1.neurons,transform=weights)
 
   
    p_true_label = nengo.Probe(true_label)
    p_layer_1 = nengo.Probe(layer1.neurons)
    #if(not full_log):
    #    nengo.Node(log)

    #############################

step_time = (presentation_time + pause_time) 

with nengo.Simulator(model,dt=0.005) as sim:
       
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

    output_spikes_num = output_spikes[num*int(presentation_time/0.005):(num+1)*int(presentation_time/0.005),:] # 0.350/0.005
    num_spikes = np.sum(output_spikes_num > 0, axis=0)

    for i in range(n_classes):
        sum_temp = 0
        count_temp = 0
        for j in range(n_neurons):
            if((neuron_class[j]) == i) : 
                sum_temp += num_spikes[j]
                count_temp +=1

        class_spikes[i] = sum_temp/count_temp



    # print(class_spikes)
    k = np.argmax(num_spikes)
    # predicted_labels.append(neuron_class[k])
    class_pred = np.argmax(class_spikes)
    predicted_labels.append(class_pred)

    true_class = labels[(num*int(presentation_time/0.005))]
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
#Ratio = Ratio + (alpha * (CRmaining / CTotal))
