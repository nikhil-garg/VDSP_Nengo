# -*- coding: utf-8 -*-
"""
Created on 4th Jan 2021

@author: Nikhil
"""

import argparse


def args():
    parser = argparse.ArgumentParser(
        description="Train a VDSP based MNIST classifier"
    )

    # Defining the model
    parser.add_argument(
        "--input_nbr",
        default=60000, 
        type=int, 
        help="Number of images to consider for training"
    )
    parser.add_argument(
        "--input_scale",
        default=1, 
        type=float, 
        help="Scaling factor for input Images"
    )
    parser.add_argument(
        "--alpha_p",
        default=0.05, 
        type=float, 
    )
    parser.add_argument(
        "--alpha_n",
        default=0.05, 
        type=float, 
    )
    parser.add_argument(
        "--beta_p",
        default=1.5, 
        type=float, 
    )
    parser.add_argument(
        "--beta_n",
        default=0.5, 
        type=float, 
    )
    parser.add_argument(
        "--tau_pre",
        default=0.03, 
        type=float, 
    )
    parser.add_argument(
        "--tau_post",
        default=0.04, 
        type=float, 
    )

    parser.add_argument(
        "--dt",
        default=0.005, 
        type=float, 
        help="Time step"
    )
    parser.add_argument(
        "--digit",
        default=4,
        type=int,
        help="The digit to consider for geting receptive field",
    )
    parser.add_argument(
        "--presentation_time",
        default=0.35,
        type=float,
        help="Presentation time of one image",
    )
    parser.add_argument(
        "--pause_time",
        default=0,
        type=float,
        help="Pause time",
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=float,
        help="Number of iterations to train for",
    )

    parser.add_argument(
        "--n_in",
        default=784,
        type=int,
        help="Number of input neurons",
    )

    parser.add_argument(
        "--amp_neuron",
        default=0.5,
        type=float,
        help="Transform from synapse to output neurons"
    )
    parser.add_argument(
        "--n_neurons",
        default=50,
        type=float,
        help="Number of output neurons",
    )
    parser.add_argument(
        "--tau_in",
        default=0.06,
        type=float,
        help="Leak constant of input neurons",
    )

    parser.add_argument(
        "--tau_out",
        default=0.06,
        type=float,
        help="Leak constant of output neurons",
    )
    parser.add_argument(
        "--lr",
        default=1,
        type=float,
        help="Learning rate of VDSP",
    )
    parser.add_argument(
        "--filename",
        default="default",
        type=str,
        help="filename of final weights",
    )

    parser.add_argument(
        "--vprog",
        default=0,
        type=float,
        help="vprog",
    )
    parser.add_argument(
        "--rate_out",
        default=20,
        type=float,
        help="Firing rate for output neuron",
    )
    parser.add_argument(
        "--rate_in",
        default=20,
        type=float,
        help="Firing rate for input neuron",
    )

    parser.add_argument(
        "--gain_out",
        default=2,
        type=float,
        help="gain for output neuron",
    )
    parser.add_argument(
        "--gain_in",
        default=2,
        type=float,
        help="gain for input neuron",
    )

    parser.add_argument(
        "--bias_out",
        default=0,
        type=float,
        help="bias for output neuron",
    )
    parser.add_argument(
        "--bias_in",
        default=0,
        type=float,
        help="bias for input neuron",
    )

    parser.add_argument(
        "--thr_out",
        default=1,
        type=float,
        help="Threshold of output layer",
    )
    parser.add_argument(
        "--inhibition_time",
        default=10,
        type=float,
        help="inhibition_time",
    )
    parser.add_argument(
        "--th_var",
        default=0,
        type=float,
        help="Variability of vth. Between 0 and 1",
    )
    parser.add_argument(
        "--vthp",
        default=0.5,
        type=float,
        help="Switching threshold of memristor",
    )
    parser.add_argument(
        "--vthn",
        default=0.5,
        type=float,
        help="Switching threshold of memristor",
    )
    parser.add_argument(
        "--weight_quant",
        default=0,
        type=float,
        help="Variability of weight update",
    )
    parser.add_argument(
        "--amp_var",
        default=0,
        type=float,
        help="Variability of Ap and An in VDSP",
    )
    parser.add_argument(
        "--amp_vth_var",
        default=0,
        type=float,
        help="Variability of Vthp, Vthn, Ap and An in VDSP",
    )
    parser.add_argument(
        "--dw_var",
        default=0,
        type=float,
        help="Variability of dW",
    )
    parser.add_argument(
        "--g_var",
        default=0,
        type=float,
        help="Variability of gmax and gmin",
    )

    parser.add_argument(
        "--gmax",
        default=0.0008,
        type=float,
    )
    parser.add_argument(
        "--gmin",
        default=0.00008,
        type=float,
    )

    parser.add_argument(
        "--log_file_path",
        default=None,
        type=str,
        help="log file path",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=float,
        help="Seed of random number generator",
    )
    parser.add_argument(
        "--vprog_increment",
        default=0,
        type=float,
        help="Vprog increment after every time step",
    )

    parser.add_argument(
        "--tau_ref_in",
        default=0.002,
        type=float,
        help="Refractory period",
    )
    parser.add_argument(
        "--tau_ref_out",
        default=0.002,
        type=float,
        help="Refractory period",
    )
    parser.add_argument(
        "--synapse_layer_1",
        default=None,
        type=float,
        help="Refractory period",
    )
    parser.add_argument(
        "--winit_max",
        default=1,
        type=float,
        help="Maximum value of winit",
    )
    parser.add_argument(
        "--voltage_clip_max",
        default=3,
        type=float,
        help="Maximum value of voltage clipping",
    )
    parser.add_argument(
        "--voltage_clip_min",
        default=-3,
        type=float,
        help="Minimum value of voltage clipping",
    )
    parser.add_argument(
        "--tau_n",
        default=1,
        type=float,
        help="Adaption leak time constant",
    )
    parser.add_argument(
        "--inc_n",
        default=0.01,
        type=float,
        help="Adaption increment",
    )
    parser.add_argument(
        "--Vapp_multiplier",
        default=0,
        type=float,
        help="Multiplier for Vapp",
    )
    parser.add_argument(
        "--noise_input",
        default=0,
        type=float,
        help="Input noise",
    )
    parser.add_argument(
        "--winit_mean",
        default=0.5,
        type=float,
        help="Winit mean",
    )
    parser.add_argument(
        "--winit_dev",
        default=0.5,
        type=float,
        help="Winit dev",
    )
    parser.add_argument(
        "--alpha",
        default=2,
        type=float,
        help="Winit dev",
    )
    parser.add_argument(
        "--dead_zone",
        default=0,
        type=float,
        help="Dead zone",
    )
    parser.add_argument(
        "--multiplicative",
        default=1,
        type=float,
        help="Whether multiplicative VDSP is enabled",
    )
    parser.add_argument(
        "--tau_stdp",
        default=0.1,
        type=float,
        help="tau of STDP",
    )



    my_args = parser.parse_args()

    return my_args
