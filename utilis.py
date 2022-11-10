import nengo
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import tensorflow as tf
import os
from nengo.dists import Choice
from datetime import datetime
import pickle
from nengo.utils.matplotlib import rasterplot

plt.rcParams.update({'figure.max_open_warning': 0})
import time

from InputData import PresentInputWithPause
# from custom_rule import CustomRule
# from custom_rule import CustomRule_prev

# import nengo_ocl

from nengo.neurons import LIFRate
# from custom_rule import CustomRule
# from custom_rule import CustomRule_prev
from nengo.params import Parameter, NumberParam, FrozenObject
from nengo.dists import Choice, Distribution, get_samples, Uniform

from nengo.utils.numpy import clip, is_array_like

from nengo.connection import LearningRule
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import LearningRuleType
from nengo.builder.learning_rules import get_pre_ens,get_post_ens
from nengo.neurons import AdaptiveLIF
from nengo.synapses import Lowpass, SynapseParam
from nengo.params import (NumberParam,Default)
from nengo.dists import Choice
from nengo.utils.numpy import clip
import numpy as np
import random
import math


def evaluation(classes,n_neurons,presentation_time,spikes_layer1_probe,label_test_filtered,dt):
    
    ConfMatrix = np.zeros((classes,n_neurons))
    labels = np.zeros(n_neurons)
    accuracy = np.zeros(n_neurons)
    total = 0
    Good = 0
    Bad = 0
    # confusion matrix
    x = 0
    for i in label_test_filtered:
            tmp = spikes_layer1_probe[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
            tmp[tmp < np.max(tmp)] = 0
            tmp[tmp != 0] = 1
            
            ConfMatrix[i] = ConfMatrix[i] + tmp

            x = x + 1
            

    Classes = dict()
    for i in range(0,n_neurons):
        Classes[i] = np.argmax(ConfMatrix[:,i])
    
    x = 0
    for i in label_test_filtered:
        correct = False
        tmp = spikes_layer1_probe[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
        tmp[tmp < np.max(tmp)] = 0
        tmp[tmp != 0] = 1

        for index,l in enumerate(tmp):
            if(l == 1):
                correct = correct or (Classes[index] == i)
        if(correct):
            Good += 1
        else:
            Bad += 1
        x = x + 1
        total += 1

    return Classes, round((Good * 100)/(Good+Bad),2)


def evaluation_v2(classes,n_neurons,presentation_time,spikes_layer1_probe_train,label_train_filtered,spikes_layer1_probe_test,label_test_filtered,dt):
    
    ConfMatrix = np.zeros((classes,n_neurons))
    labels = np.zeros(n_neurons)
    accuracy = np.zeros(n_neurons)
    total = 0
    Good = 0
    Bad = 0
    # confusion matrix
    x = 0
    for i in label_train_filtered:
            tmp = spikes_layer1_probe_train[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
            tmp[tmp < np.max(tmp)] = 0
            tmp[tmp != 0] = 1
            
            ConfMatrix[i] = ConfMatrix[i] + tmp

            x = x + 1
            

    Classes = dict()
    for i in range(0,n_neurons):
        Classes[i] = np.argmax(ConfMatrix[:,i])
    
    x = 0
    for i in label_test_filtered:
        correct = False
        tmp = spikes_layer1_probe_test[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
        tmp[tmp < np.max(tmp)] = 0
        tmp[tmp != 0] = 1

        for index,l in enumerate(tmp):
            if(l == 1):
                correct = correct or (Classes[index] == i)
        if(correct):
            Good += 1
        else:
            Bad += 1
        x = x + 1
        total += 1

    return round((Good * 100)/(Good+Bad),2)



class MyLIF_in(LIFRate):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    state = {
        "voltage": Uniform(low=0, high=1),
        "refractory_time": Choice([0]),
    }
    spiking = True

    min_voltage = NumberParam("min_voltage", high=0)

    def __init__(
        self, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1, initial_state=None
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.min_voltage = min_voltage

    def step(self, dt, J, output, voltage, refractory_time):
        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1.8
        output[:] = spiked_mask * (self.amplitude / dt)

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = -1.8 #reset voltage
        refractory_time[spiked_mask] = self.tau_ref + t_spike

def plan_MyLIF_in(
    queue,
    dt,
    J,
    V,
    W,
    outS,
    ref,
    tau,
    amp,
    N=None,
    tau_n=None,
    inc_n=None,
    upsample=1,
    fastlif=False,
    **kwargs
):
    adaptive = N is not None
    assert J.ctype == "float"
    for x in [V, W, outS]:
        assert x.ctype == J.ctype
    adaptive = False
    fastlif = False
    inputs = dict(J=J, V=V, W=W)
    outputs = dict(outV=V, outW=W, outS=outS)
    parameters = dict(tau=tau, ref=ref, amp=amp)
    if adaptive:
        assert all(x is not None for x in [N, tau_n, inc_n])
        assert N.ctype == J.ctype
        inputs.update(dict(N=N))
        outputs.update(dict(outN=N))
        parameters.update(dict(tau_n=tau_n, inc_n=inc_n))

    dt = float(dt)
    textconf = dict(
        type=J.ctype,
        dt=dt,
        upsample=upsample,
        adaptive=adaptive,
        dtu=dt / upsample,
        dtu_inv=upsample / dt,
        dt_inv=1 / dt,
        fastlif=fastlif,
    )
    decs = """
        char spiked;
        ${type} dV;
        const ${type} V_threshold = 1;
        const ${type} dtu = ${dtu}, dtu_inv = ${dtu_inv}, dt_inv = ${dt_inv};
% if adaptive:
        const ${type} dt = ${dt};
% endif
%if fastlif:
        const ${type} delta_t = dtu;
%else:
        ${type} delta_t;
%endif
        """
    # TODO: could precompute -expm1(-dtu / tau)
    text = """
        spiked = 0;
% for ii in range(upsample):
        W -= dtu;
% if not fastlif:
        delta_t = (W > dtu) ? 0 : (W < 0) ? dtu : dtu - W;
% endif
% if adaptive:
        dV = -expm1(-delta_t / tau) * (J - N - V);
% else:
        dV = -expm1(-delta_t / tau) * (J - V);
% endif
        V += dV;
% if fastlif:
        if (V < 0 || W > dtu)
            V = 0;
        else if (W >= 0)
            V *= 1 - W * dtu_inv;
% endif
        if (V > V_threshold) {
% if fastlif:
            const ${type} overshoot = dtu * (V - V_threshold) / dV;
            W = ref - overshoot + dtu;
% else:
            const ${type} t_spike = dtu + tau * log1p(
                -(V - V_threshold) / (J - V_threshold));
            W = ref + t_spike;
% endif
            V = 0;
            spiked = 1;
        }
% if not fastlif:
         else if (V < 0) {
            V = 0;
        }
% endif
% endfor
        outV = V;
        outW = W;
        outS = (spiked) ? amp*dt_inv : 0;
% if adaptive:
        outN = N + (dt / tau_n) * (inc_n * outS - N);
% endif
        """
    decs = as_ascii(Template(decs, output_encoding="ascii").render(**textconf))
    text = as_ascii(Template(text, output_encoding="ascii").render(**textconf))
    cl_name = "cl_alif" if adaptive else "cl_lif"
    return _plan_template(
        queue,
        cl_name,
        text,
        declares=decs,
        inputs=inputs,
        outputs=outputs,
        parameters=parameters,
        **kwargs,
    )

class MyLIF_in_v2(LIFRate):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    state = {
        "voltage": Uniform(low=0, high=1),
        "refractory_time": Choice([0]),
    }
    spiking = True

    min_voltage = NumberParam("min_voltage", high=0)

    def __init__(
        self, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1, initial_state=None
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.min_voltage = min_voltage

    def step(self, dt, J, output, voltage, refractory_time):
        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        output[:] = spiked_mask * (self.amplitude / dt)

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = -1 #reset voltage
        refractory_time[spiked_mask] = self.tau_ref + t_spike

        
class MyLIF_out(LIFRate):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    state = {
        "voltage": Uniform(low=0, high=1),
        "refractory_time": Choice([0]),
    }
    spiking = True

    min_voltage = NumberParam("min_voltage", high=0)

    def __init__(
        self, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1, initial_state=None, inhib=[]
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.min_voltage = min_voltage
        self.inhib = inhib

    def step(self, dt, J, output, voltage, refractory_time):
        # look these up once to avoid repeated parameter accesses

        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        output[:] = spiked_mask * (self.amplitude / dt)
        
        if(np.sum(output)!=0):
            voltage[voltage != np.max(voltage)] = 0 #WTA
            
        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = -1
        refractory_time[spiked_mask] = self.tau_ref + t_spike

# def fun_post_baseline(X,
#        alpha=1,lr=1,dead_zone=0,multiplicative=1
#        ): 
    
#     w, vmem = X
#     vapp = -vmem
#     # multiplicative = 1
        
#     f_pot = (np.exp(-alpha*(w))*(-w+ 1))*multiplicative
#     f_dep = (np.exp(alpha*(w-1))*w)*multiplicative
    
#     cond_pot = vapp > dead_zone
#     cond_dep = vapp < -dead_zone
    
#     exp_vapp = np.exp(vapp-dead_zone)
#     g_pot = exp_vapp-1
#     g_dep = (1/exp_vapp)-1

#     # g_pot = np.exp(vapp)-1
#     # g_dep = np.exp(-vapp)-1

#     dW = (cond_pot*f_pot*g_pot  -  cond_dep*f_dep*g_dep)*lr
#     return dW

def fun_post_baseline(X,
       alpha=1,lr=1
       ): 
    
    w, vmem = X
        
    f_pot = 1-w
    f_dep = w
    
    cond_pot = vmem < 0
    cond_dep = vmem > 0
    
    exp_vapp = np.exp(-vmem)

    g_pot = exp_vapp-1
    g_dep = (1/exp_vapp)-1


    dW = (cond_pot*f_pot*g_pot  -  cond_dep*f_dep*g_dep)*lr
    return dW

def fun_post_baseline_lite(X,
       lr=1
       ): 
    
    w, vmem = X
    vapp = -vmem

    cond_pot = vapp > 0
    cond_dep = vapp < 0
    
    exp_vapp = np.exp(vapp)
    g_pot = exp_vapp-1
    g_dep = (1/exp_vapp)-1


    dW = (cond_pot*g_pot  -  cond_dep*g_dep)*lr
    return dW

class CustomRule_post_baseline(nengo.Process):
   
    def __init__(self,winit_min=0, winit_max=1, sample_distance = 1, lr=6.0e-05, alpha=2):
        
        self.signal_vmem_pre = None
        self.signal_out_post = None
        self.winit_min = winit_min
        self.winit_max = winit_max
        self.alpha = alpha
        self.sample_distance = sample_distance
        self.lr = lr
        # self.multiplicative = multiplicative
        # self.dead_zone = dead_zone
        self.history = [0]        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
                        
            post_out = self.signal_out_post
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   
            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))
            
            self.w = np.clip((self.w + dt*(fun_post_baseline((self.w,vmem),self.alpha,self.lr))*post_out_matrix), 0, 1)
            self.history[0] = self.w.copy()
            return np.dot(self.w, x*dt)
        
        return step   

    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal


class CustomRule_post_baseline_lite(nengo.Process):
   
    def __init__(self,winit_min=0, winit_max=1, sample_distance = 1, lr=6.0e-05):
        
        self.signal_vmem_pre = None
        self.signal_out_post = None
        self.winit_min = winit_min
        self.winit_max = winit_max
        # self.alpha = alpha
        self.sample_distance = sample_distance
        self.lr = lr
        # self.multiplicative = multiplicative
        # self.dead_zone = dead_zone
        self.history = [0]        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
                        
            post_out = self.signal_out_post
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   
            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))
            
            self.w = np.clip((self.w + dt*(fun_post_baseline((self.w,vmem),self.lr))*post_out_matrix), 0, 1)
            self.history[0] = self.w.copy()
            return np.dot(self.w, x*dt)
        
        return step   

    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal


def fun_post(X,
       alphap=1,alphan=5,Ap=4000,An=4000,eta=1
       ): 
    
    w, vmem, vprog, vthp,vthn = X
    # vthp=0.16
    # vthn=0.15
    # vprog=0
    xp=0.3
    xn=0.5 
    vapp = vprog-vmem
    
    cond_pot_fast = w<xp
    cond_pot_slow = 1-cond_pot_fast
    
    cond_dep_fast = w>(1-xn)
    cond_dep_slow = 1-cond_dep_fast
    
    f_pot = (np.exp(-alphap*(w-xp))*((xp-w)/(1-xp) + 1))*cond_pot_slow + cond_pot_fast
    f_dep = (np.exp(alphan*(w+xn-1))*w/(1-xn))*cond_dep_slow + cond_dep_fast
    
    cond_pot = vapp > vthp
    cond_dep = vapp < -vthn
    
    g_pot = Ap*(np.exp(vapp)-np.exp(vthp))
    g_dep = -An*(np.exp(-vapp)-np.exp(vthn))

    dW = (cond_pot*f_pot*g_pot  +  cond_dep*f_dep*g_dep)*eta
    return dW

popt = np.array((1.00006690e+00, 5.00098722e+00, 1.27251859e-05, 1.27251790e-05,
       6.28659913e+00))
#The above popt is for Tpw = 20n, maximum dw=0.0001. Use lr=1




def fun_post_var(X,
       alphap=1,alphan=5,Ap=4000,An=4000,eta=1
       ): 
    
    w, vmem, vprog, vthp,vthn,var_amp_1,var_amp_2  = X
    # vthp=0.16
    # vthn=0.15
    # vprog=0
    xp=0.3
    xn=0.5 
    vapp = vprog-vmem

    Ap = var_amp_1*Ap
    An = var_amp_2*An
    
    cond_pot_fast = w<xp
    cond_pot_slow = 1-cond_pot_fast
    
    cond_dep_fast = w>(1-xn)
    cond_dep_slow = 1-cond_dep_fast
    
    f_pot = (np.exp(-alphap*(w-xp))*((xp-w)/(1-xp) + 1))*cond_pot_slow + cond_pot_fast
    f_dep = (np.exp(alphan*(w+xn-1))*w/(1-xn))*cond_dep_slow + cond_dep_fast
    
    cond_pot = vapp > vthp
    cond_dep = vapp < -vthn
    
    g_pot = Ap*(np.exp(vapp)-np.exp(vthp))
    g_dep = -An*(np.exp(-vapp)-np.exp(vthn))

    dW = (cond_pot*f_pot*g_pot  +  cond_dep*f_dep*g_dep)*eta
    return dW

popt = np.array((1.00220687e+00,  5.01196597e+00, -3.54137489e-03, -3.54157996e-03,
       -2.25853150e-01))



def fun_post_tio2(X,
       alphap=1,alphan=5,Ap=4000,An=4000,eta=1,
       # a1=1,a2=1,a3=1,a4=1
       ): 
    
    w, vmem, vprog, vthp, vthn, voltage_clip_max, voltage_clip_min, Vapp_multiplier = X
    # vthp=0.5
    # vthn=0.5
    # vprog=0
    xp=0.01
    xn=0.01
    
    # vapp = (vprog-vmem)*(1+w*Vapp_multiplier)
    vapp = (vprog-vmem)*Vapp_multiplier

    vapp = np.clip(vapp, voltage_clip_min, voltage_clip_max)
    
    cond_pot_fast = w<xp
    cond_pot_slow = 1-cond_pot_fast
    
    cond_dep_fast = w>(1-xn)
    cond_dep_slow = 1-cond_dep_fast
    
    f_pot = cond_pot_fast + cond_pot_slow*(np.exp(-alphap*(w-xp))*((xp-w)/(1-xp) + 1))
    f_dep = (np.exp(alphan*(w+xn-1))*w/(1-xn))*cond_dep_slow + cond_dep_fast
    
    cond_pot = vapp > vthp
    cond_dep = vapp < -vthn
    
    g_pot = Ap*(np.exp(vapp)-np.exp(vthp))
    g_dep = -An*(np.exp(-vapp)-np.exp(vthn))

    dW = (cond_pot*f_pot*g_pot  +  cond_dep*f_dep*g_dep)*eta
    return dW

# parameter for TiO2 with 0.5 threshold and 77.07 accuracy
popt_tio2 = np.array((1.62708935, 2.1204144 , 0.044, 0.07223655, 0.95411709))

# popt_tio2 = np.array((0.86066859, 1.28831255, 0.44703269, 0.21166331, 0.80906049))

def fun_post_tio2_var(X,
       alphap=1,alphan=5,Ap=4000,An=4000,eta=1,
       # a1=1,a2=1,a3=1,a4=1
       ): 
    
    w, vmem, vprog, vthp, vthn, var_amp_1, var_amp_2, voltage_clip_max, voltage_clip_min = X
    # vthp=0.5
    # vthn=0.5
    # vprog=0
    xp=0.01
    xn=0.01
    Ap = var_amp_1*Ap
    An = var_amp_2*An

    vapp = vprog-vmem
    vapp = np.clip(vapp, voltage_clip_min, voltage_clip_max)
    
    cond_pot_fast = w<xp
    cond_pot_slow = 1-cond_pot_fast
    
    cond_dep_fast = w>(1-xn)
    cond_dep_slow = 1-cond_dep_fast
    
    f_pot = cond_pot_fast + cond_pot_slow*(np.exp(-alphap*(w-xp))*((xp-w)/(1-xp) + 1))
    f_dep = (np.exp(alphan*(w+xn-1))*w/(1-xn))*cond_dep_slow + cond_dep_fast
    
    cond_pot = vapp > vthp
    cond_dep = vapp < -vthn
    
    g_pot = Ap*(np.exp(vapp)-np.exp(vthp))
    g_dep = -An*(np.exp(-vapp)-np.exp(vthn))

    dW = (cond_pot*f_pot*g_pot  +  cond_dep*f_dep*g_dep)*eta
    return dW

def fun_post_tio2_var_th(X,
       alphap=1,alphan=5,Ap=4000,An=4000,eta=1,
       # a1=1,a2=1,a3=1,a4=1
       ): 
    
    w, vmem, vprog, vthp, vthn, var_th_1, var_th_2, voltage_clip_max, voltage_clip_min = X
    # vthp=0.5
    # vthn=0.5
    # vprog=0
    xp=0.01
    xn=0.01
    vthp = var_th_1*vthp
    vthn = var_th_2*vthn

    vapp = vprog-vmem
    vapp = np.clip(vapp, voltage_clip_min, voltage_clip_max)
    
    cond_pot_fast = w<xp
    cond_pot_slow = 1-cond_pot_fast
    
    cond_dep_fast = w>(1-xn)
    cond_dep_slow = 1-cond_dep_fast
    
    f_pot = cond_pot_fast + cond_pot_slow*(np.exp(-alphap*(w-xp))*((xp-w)/(1-xp) + 1))
    f_dep = (np.exp(alphan*(w+xn-1))*w/(1-xn))*cond_dep_slow + cond_dep_fast
    
    cond_pot = vapp > vthp
    cond_dep = vapp < -vthn
    
    g_pot = Ap*(np.exp(vapp)-np.exp(vthp))
    g_dep = -An*(np.exp(-vapp)-np.exp(vthn))

    dW = (cond_pot*f_pot*g_pot  +  cond_dep*f_dep*g_dep)*eta
    return dW

def fun_post_tio2_var_v2(X,
       alphap=1,alphan=5,Ap=4000,An=4000,eta=1,
       # a1=1,a2=1,a3=1,a4=1
       ): 
    
    w, vmem, vprog, vthp, vthn, var_amp_1, var_amp_2, var_vthp, var_vthn, voltage_clip_max, voltage_clip_min = X
    # vthp=0.5
    # vthn=0.5
    # vprog=0
    xp=0.01
    xn=0.01
    Ap = var_amp_1*Ap
    An = var_amp_2*An
    vthp = var_vthp*vthp
    vthn = var_vthn*vthn

    vapp = vprog-vmem
    vapp = np.clip(vapp, voltage_clip_min, voltage_clip_max)
    
    cond_pot_fast = w<xp
    cond_pot_slow = 1-cond_pot_fast
    
    cond_dep_fast = w>(1-xn)
    cond_dep_slow = 1-cond_dep_fast
    
    f_pot = cond_pot_fast + cond_pot_slow*(np.exp(-alphap*(w-xp))*((xp-w)/(1-xp) + 1))
    f_dep = (np.exp(alphan*(w+xn-1))*w/(1-xn))*cond_dep_slow + cond_dep_fast
    
    cond_pot = vapp > vthp
    cond_dep = vapp < -vthn
    
    g_pot = Ap*(np.exp(vapp)-np.exp(vthp))
    g_dep = -An*(np.exp(-vapp)-np.exp(vthn))

    dW = (cond_pot*f_pot*g_pot  +  cond_dep*f_dep*g_dep)*eta
    return dW

class CustomRule_post_v2_tio2(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.5,vthn=0.5,gmax=0.0008,gmin=0.00008,vprog_increment=0,voltage_clip_max=None,voltage_clip_min=None,Vapp_multiplier=0):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.gmax=gmax
        self.gmin = gmin
        
        self.history = [0]

        self.voltage_clip_min=voltage_clip_min
        self.voltage_clip_max=voltage_clip_max
        self.vprog_increment=vprog_increment
        self.Vapp_multiplier = Vapp_multiplier

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            self.w = np.clip((self.w + dt*(fun_post_tio2((self.w,vmem, self.vprog, self.vthp,self.vthn,self.voltage_clip_max,self.voltage_clip_min,self.Vapp_multiplier),*popt_tio2))*post_out_matrix*self.lr), 0, 1)

            post_spiked = post_out_matrix*dt
            self.vprog += post_spiked*self.vprog_increment
            self.vprog = np.clip(self.vprog, None, 0)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w

            return np.dot((self.w*(self.gmax-self.gmin)) + self.gmin, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal


class CustomRule_post_v2_tio2_gaussian(nengo.Process):
   
    def __init__(self, vprog=0,winit_mean=5, winit_dev=0.5, sample_distance = 1, lr=1,vthp=0.5,vthn=0.5,gmax=0.0008,gmin=0.00008,vprog_increment=0,voltage_clip_max=None,voltage_clip_min=None,Vapp_multiplier=0):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_mean = winit_mean
        self.winit_dev = winit_dev
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.gmax=gmax
        self.gmin = gmin
        
        self.history = [0]

        self.voltage_clip_min=voltage_clip_min
        self.voltage_clip_max=voltage_clip_max
        self.vprog_increment=vprog_increment
        self.Vapp_multiplier = Vapp_multiplier

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.normal(self.winit_mean, self.winit_dev, (shape_out[0], shape_in[0]))
        self.w = np.clip(self.w,0,1)
        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            self.w = np.clip((self.w + dt*(fun_post_tio2((self.w,vmem, self.vprog, self.vthp,self.vthn,self.voltage_clip_max,self.voltage_clip_min,self.Vapp_multiplier),*popt_tio2))*post_out_matrix*self.lr), 0, 1)

            post_spiked = post_out_matrix*dt
            self.vprog += post_spiked*self.vprog_increment
            self.vprog = np.clip(self.vprog, None, 0)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w

            return np.dot((self.w*(self.gmax-self.gmin)) + self.gmin, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal

class CustomRule_post_v3_tio2(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.5,vthn=0.5,gmax=0.0008,gmin=0.00008,vprog_increment=0,voltage_clip_max=None,voltage_clip_min=None,Vapp_multiplier=0):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.gmax=gmax
        self.gmin = gmin
        
        self.history = [0]
        self.current_weight = [0]

        self.voltage_clip_min=voltage_clip_min
        self.voltage_clip_max=voltage_clip_max
        self.vprog_increment=vprog_increment
        self.Vapp_multiplier = Vapp_multiplier

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            self.w = np.clip((self.w + dt*(fun_post_tio2((self.w,vmem, self.vprog, self.vthp,self.vthn,self.voltage_clip_max,self.voltage_clip_min,self.Vapp_multiplier),*popt_tio2))*post_out_matrix*self.lr), 0, 1)

            post_spiked = post_out_matrix*dt
            self.vprog += post_spiked*self.vprog_increment
            self.vprog = np.clip(self.vprog, None, 0)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()

            self.current_weight[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w

            return np.dot((self.w*(self.gmax-self.gmin)) + self.gmin, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal



class CustomRule_post_v2(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.16,vthn=0.15,gmax=0.0085,gmin=0.0000085):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.gmax=gmax
        self.gmin = gmin
        
        self.history = [0]

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            self.w = np.clip((self.w + dt*(fun_post((self.w,vmem, self.vprog, self.vthp,self.vthn),*popt))*post_out_matrix*self.lr), 0.001, 1)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w

            return np.dot((self.w*(self.gmax-self.gmin)) + self.gmin, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal

class CustomRule_post_v2_history(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.16,vthn=0.15):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        
        self.history = []
        self.update_history = []

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            dw = dt*(fun_post((self.w,vmem, self.vprog, self.vthp,self.vthn),*popt))*post_out_matrix*self.lr
            self.w = np.clip((self.w + dw), 0.001, 1)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            # self.history[0] = self.w.copy()
            self.history.append(self.w.copy())
            self.update_history.append(dw)
            # self.history = self.history[-2:]
            # self.history = self.w
            
            return np.dot(self.w, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal



#For C2C variability

class CustomRule_post_v3(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.16,vthn=0.15,var_dw=0,gmax=0.0085, gmin=0.0000085):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        self.var_dw = var_dw

        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.gmin=gmin
        self.gmax = gmax
        
        self.history = [0]

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))
            
            np.random.seed(int(t))
            random.seed(int(t)) 
            
            random_matrix = np.random.normal(0.0, 1.0, (shape_out[0],shape_in[0]))
            var_matrix = (random_matrix*self.var_dw)+1

            self.w = np.clip((self.w + dt*(fun_post((self.w,vmem, self.vprog, self.vthp,self.vthn),*popt))*post_out_matrix*self.lr*var_matrix), 0.001, 1)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w
            
            return np.dot((self.w*(self.gmax-self.gmin)) + self.gmin, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal



#For variability in Ap and An 
class CustomRule_post_v4(nengo.Process):
    #var is the matrix with random numbers to be multiplied with Ap and An. var=1 : no variability
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.16,vthn=0.15,var_amp_1= 1,var_amp_2=1,gmax=0.0085, gmin=0.0000085):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.var_amp_1 = var_amp_1
        self.var_amp_2 = var_amp_2
        self.gmin=gmin
        self.gmax = gmax
        
        self.history = [0]

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            self.w = np.clip((self.w + dt*(fun_post_var((self.w,vmem, self.vprog, self.vthp,self.vthn,self.var_amp_1,self.var_amp_2),*popt))*post_out_matrix*self.lr), 0.001, 1)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w
            
            return np.dot((self.w*(self.gmax-self.gmin)) + self.gmin, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal

#C2C variability in Ap, and An for TiO2 based devices
class CustomRule_post_v4_tio2(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.5,vthn=0.5,var_amp=0,gmax=0.0008,gmin=0.00008,voltage_clip_max=None,voltage_clip_min=None):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max

        self.voltage_clip_min=voltage_clip_min
        self.voltage_clip_max=voltage_clip_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.gmax=gmax
        self.gmin = gmin
        self.var_amp=var_amp
        
        self.history = [0]

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   
            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            random_matrix = np.random.normal(0.0, 1.0, (shape_out[0],shape_in[0])) #between -1 to 1 of shape W
            var_amp_matrix_1 = 1 + (random_matrix*self.var_amp)
            random_matrix = np.random.normal(0.0, 1.0, (shape_out[0],shape_in[0])) #between -1 to 1 of shape W
            var_amp_matrix_2 = 1 + (random_matrix*self.var_amp)

            self.w = np.clip((self.w + dt*(fun_post_tio2_var((self.w,vmem, self.vprog, self.vthp,self.vthn,var_amp_matrix_1,var_amp_matrix_2, self.voltage_clip_max, self.voltage_clip_min),*popt_tio2))*post_out_matrix*self.lr), 0, 1)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w

            return np.dot((self.w*(self.gmax-self.gmin)) + self.gmin, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal

class CustomRule_post_v5_tio2(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.5,vthn=0.5,var_th= 0,gmax=0.0008,gmin=0.00008,voltage_clip_max=None,voltage_clip_min=None):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max

        self.voltage_clip_min=voltage_clip_min
        self.voltage_clip_max=voltage_clip_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.gmax=gmax
        self.gmin = gmin
        self.var_th=var_th
        
        self.history = [0]

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            random_matrix = np.random.normal(0.0, 1.0, (shape_out[0],shape_in[0])) #between -1 to 1 of shape W
            var_th_matrix_1 = 1 + (random_matrix*self.var_th)
            random_matrix = np.random.normal(0.0, 1.0, (shape_out[0],shape_in[0])) #between -1 to 1 of shape W
            var_th_matrix_2 = 1 + (random_matrix*self.var_th)



            post_out = self.signal_out_post
            
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            self.w = np.clip((self.w + dt*(fun_post_tio2_var_th((self.w,vmem, self.vprog, self.vthp,self.vthn,var_th_matrix_1,var_th_matrix_2, self.voltage_clip_max, self.voltage_clip_min),*popt_tio2))*post_out_matrix*self.lr), 0, 1)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w

            return np.dot((self.w*(self.gmax-self.gmin)) + self.gmin, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal

class CustomRule_post_v6_tio2(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.5,vthn=0.5,var_amp_th=0,gmax=0.0008,gmin=0.00008, voltage_clip_max=None, voltage_clip_min=None):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.gmax=gmax
        self.gmin = gmin
        self.var_amp_th=var_amp_th


        self.voltage_clip_min=voltage_clip_min
        self.voltage_clip_max=voltage_clip_max
        
        self.history = [0]

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            random_matrix = np.random.normal(0.0, 1.0, (shape_out[0],shape_in[0])) #between -1 to 1 of shape W
            var_th_matrix_1 = 1 + (random_matrix*self.var_amp_th)
            random_matrix = np.random.normal(0.0, 1.0, (shape_out[0],shape_in[0])) #between -1 to 1 of shape W
            var_th_matrix_2 = 1 + (random_matrix*self.var_amp_th)

            random_matrix = np.random.normal(0.0, 1.0, (shape_out[0],shape_in[0])) #between -1 to 1 of shape W
            var_amp_matrix_1 = 1 + (random_matrix*self.var_amp_th)
            random_matrix = np.random.normal(0.0, 1.0, (shape_out[0],shape_in[0])) #between -1 to 1 of shape W
            var_amp_matrix_2 = 1 + (random_matrix*self.var_amp_th)



            post_out = self.signal_out_post
            
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            self.w = np.clip((self.w + dt*(fun_post_tio2_var_v2((self.w,vmem, self.vprog, self.vthp,self.vthn,var_amp_matrix_1,var_amp_matrix_2,var_th_matrix_1,var_th_matrix_2, self.voltage_clip_max, self.voltage_clip_min),*popt_tio2))*post_out_matrix*self.lr), 0, 1)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w

            return np.dot((self.w*(self.gmax-self.gmin)) + self.gmin, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal


class CustomRule_post_v2_tio2_history(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.5,vthn=0.5,gmax=0.0008,gmin=0.00008,vprog_increment=0,voltage_clip_max=None,voltage_clip_min=None,Vapp_multiplier=0):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.gmax=gmax
        self.gmin = gmin
        
        self.history = []

        self.voltage_clip_min=voltage_clip_min
        self.voltage_clip_max=voltage_clip_max
        self.vprog_increment=vprog_increment
        self.Vapp_multiplier = Vapp_multiplier

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            # vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(self.signal_vmem_pre, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            self.w = np.clip((self.w + dt*(fun_post_tio2((self.w,vmem, self.vprog, self.vthp,self.vthn,self.voltage_clip_max,self.voltage_clip_min,self.Vapp_multiplier),*popt_tio2))*post_out_matrix*self.lr), 0, 1)

            post_spiked = post_out_matrix*dt
            self.vprog += post_spiked*self.vprog_increment
            self.vprog = np.clip(self.vprog, None, 0)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            # self.history[0] = self.w.copy()
            self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w

            return np.dot((self.w*(self.gmax-self.gmin)) + self.gmin, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal



import numpy as np
from nengo.builder.builder import Builder
from nengo.builder.learning_rules import (build_or_passthrough, get_post_ens,
                                          get_pre_ens)
from nengo.builder.operator import Copy, DotInc, Operator, Reset
from nengo.learning_rules import LearningRuleType, _remove_default_post_synapse
from nengo.params import Default, NdarrayParam, NumberParam
from nengo.synapses import Lowpass, SynapseParam


class VLR(LearningRuleType):
    """
    See the Nengo codebase
    (https://github.com/nengo/nengo/blob/master/nengo/learning_rules.py)
    for documentation and examples of how to construct this class, and what the super
    class constructor values are.
    """

    modifies = "weights"
    probeable = ("pre_voltages", "post_activities", "post_filtered","weights")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    vprog = NumberParam("vprog", readonly=True, default=-0.6)
    vthp = NumberParam("vthp", readonly=True, default=0.16)
    vthn = NumberParam("vthn", readonly=True, default=0.15)

    def __init__(
        self,
        learning_rate=Default,
        post_synapse=Default,
        vprog=Default,
        vthp=Default,
        vthn=Default
    ):
        super().__init__(learning_rate, size_in=0)
        self.post_synapse = post_synapse
        self.vprog = vprog
        self.vthp = vthp
        self.vthn = vthn


class SimVLR(Operator):
    """
    See the Nengo codebase
    (https://github.com/nengo/nengo/blob/master/nengo/builder/learning_rules.py)
    for the other examples of learning rule operators.
    """

    def __init__(self, pre_voltages, post_filtered,weights, delta, learning_rate,vprog,vthp,vthn, tag=None):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate
        self.vprog = vprog
        self.vthp = vthp
        self.vthn = vthn

        # Define what this operator sets, increments, reads and updates
        # See (https://github.com/nengo/nengo/blob/master/nengo/builder/operator.py)
        # for some other example operators
        self.sets = []
        self.incs = []
        self.reads = [pre_voltages, post_filtered, weights]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_voltages(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]
    
    @property
    def weights(self):
        return self.reads[2]

    @property
    def _descstr(self):
        return f"pre={self.pre_voltages}, post={self.post_filtered} -> {self.delta}"

    def make_step(self, signals, dt, rng):
        # Get signals from model signal dictionary
        pre_voltages = signals[self.pre_voltages]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        weights = signals[self.weights]
        
        pre_voltages = np.reshape(pre_voltages, (1, delta.shape[1]))
        post_filtered = np.reshape(post_filtered, (delta.shape[0], 1))

        def step_vlr():
            # Put learning rule logic here
            
            delta[...] = post_filtered*dt*fun_post((weights,pre_voltages,self.vprog,self.vthp,self.vthn),*popt)*self.learning_rate

        return step_vlr


@Builder.register(VLR)  # Register the function below with the Nengo builder
def build_vlr(model, vlr, rule):
    """
    See the Nengo codebase
    (https://github.com/nengo/nengo/blob/master/nengo/builder/learning_rules.py#L594)
    for the documentation for this function.
    """

    # Extract necessary signals and objects from the model and learning rule
    conn = rule.connection
    pre_voltages = model.sig[get_pre_ens(conn).neurons]["voltage"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    weights = model.sig[conn]["weights"]
    
    post_filtered = build_or_passthrough(model, vlr.post_synapse, post_activities)
#     post_filtered = post_activities

    # Instantiate and add the custom learning rule operator to the Nengo model op graph
    model.add_op(
        SimVLR(
            pre_voltages,
            post_filtered,
            weights,
            model.sig[rule]["delta"],
            learning_rate=vlr.learning_rate,
            vprog = vlr.vprog,
            vthp = vlr.vthp,
            vthn = vlr.vthn
            
        )
    )

    # Expose these signals for probes
    model.sig[rule]["pre_voltages"] = pre_voltages
    model.sig[rule]["post_activities"] = post_activities
    model.sig[rule]["post_filtered"] = post_filtered


#create new neuron type STDPLIF 
def build_or_passthrough(model, obj, signal):
    """Builds the obj on signal, or returns the signal if obj is None."""
    return signal if obj is None else model.build(obj, signal)

#---------------------------------------------------------------------
# Neuron Model declaration 
#---------------------------------------------------------------------

#create new neuron type STDPLIF 

# class STDPLIF(AdaptiveLIF):
#     probeable = ('spikes', 'voltage', 'refractory_time','adaptation','inhib') #,'inhib'
    
#     def __init__(self, spiking_threshold =1, inhibition_time=10,inhib=[],T = 0.0, **lif_args): # inhib=[],T = 0.0
#         super(STDPLIF, self).__init__(**lif_args)
#         # neuron args (if you have any new parameters other than gain
#         # an bais )
#         self.inhib = inhib
#         self.T = T
#         self.spiking_threshold=spiking_threshold
#         self.inhibition_time=inhibition_time
#     @property
#     def _argreprs(self):
#         args = super(STDPLIF, self)._argreprs
#         print("argreprs")
#         return args

#     # dt : timestamps 
#     # J : Input currents associated with each neuron.
#     # output : Output activities associated with each neuron.
#     def step(self, dt, J, output, voltage, refractory_time, adaptation,inhib):#inhib

#         self.T += dt
        
#         # if(np.max(J) !=0):
#         #     J = np.divide(J,np.max(J)) * 2

#         n = adaptation
        
#         J = J - n
#         # ----------------------------

#         # look these up once to avoid repeated parameter accesses
#         tau_rc = self.tau_rc
#         min_voltage = self.min_voltage

#         # reduce all refractory times by dt
#         refractory_time -= dt

#         # compute effective dt for each neuron, based on remaining time.
#         # note that refractory times that have completed midway into this
#         # timestep will be given a partial timestep, and moreover these will
#         # be subtracted to zero at the next timestep (or reset by a spike)
#         delta_t = clip((dt - refractory_time), 0, dt)

#         # update voltage using discretized lowpass filter
#         # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
#         # J is constant over the interval [t, t + dt)
#         voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

#         # determine which neurons spiked (set them to 1/dt, else 0)
#         spiked_mask = (voltage > self.spiking_threshold)
#         output[:] = spiked_mask * (self.amplitude / dt)
#         output[voltage != np.max(voltage)] = 0  
#         if(np.sum(output) != 0):
#             voltage[voltage != np.max(voltage)] = 0 
#             # inhib[(voltage != np.max(voltage)) & (inhib == 0)] = 2
#             inhib[(voltage != np.max(voltage)) & (inhib == 0)] = self.inhibition_time/(dt*1000)
#         #print("voltage : ",voltage)
#         voltage[inhib != 0] = 0
#         J[inhib != 0] = 0
#         # set v(0) = 1 and solve for t to compute the spike time
#         t_spike = dt + tau_rc * np.log1p(
#             -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
#         )
#         # set spiked voltages to zero, refractory times to tau_ref, and
#         # rectify negative voltages to a floor of min_voltage
#         voltage[voltage < min_voltage] = min_voltage
#         voltage[spiked_mask] = -1 #Reset voltage
#         voltage[refractory_time > 0] = -1 #Refractory voltage
#         refractory_time[spiked_mask] = self.tau_ref + t_spike
#         # ----------------------------

#         n += (dt / self.tau_n) * (self.inc_n * output - n)

#         #AdaptiveLIF.step(self, dt, J, output, voltage, refractory_time, adaptation)
#         inhib[inhib != 0] += - 1
#         #J[...] = 0
#         #output[...] = 0
class STDPLIF(AdaptiveLIF):
    probeable = ('spikes', 'voltage', 'refractory_time','adaptation','inhib') #,'inhib'
    
    def __init__(self, spiking_threshold =1, inhibition_time=10,inhib=[],T = 0.0, **lif_args): # inhib=[],T = 0.0
        super(STDPLIF, self).__init__(**lif_args)
        # neuron args (if you have any new parameters other than gain
        # an bais )
        self.inhib = inhib
        self.T = T
        self.spiking_threshold=spiking_threshold
        self.inhibition_time=inhibition_time
    @property
    def _argreprs(self):
        args = super(STDPLIF, self)._argreprs
        print("argreprs")
        return args

    # dt : timestamps 
    # J : Input currents associated with each neuron.
    # output : Output activities associated with each neuron.
    def step(self, dt, J, output, voltage, refractory_time, adaptation,inhib):#inhib

        self.T += dt
        
        # if(np.max(J) !=0):
        #     J = np.divide(J,np.max(J)) * 2

        n = adaptation
        
        J = J - n
        # ----------------------------

        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = (voltage > self.spiking_threshold)
        output[:] = spiked_mask * (self.amplitude / dt)
        output[voltage != np.max(voltage)] = 0  
        if(np.sum(output) != 0):
            voltage[voltage != np.max(voltage)] = 0 
            # inhib[(voltage != np.max(voltage)) & (inhib == 0)] = 2
            inhib[(voltage != np.max(voltage)) & (inhib == 0)] = self.inhibition_time/(dt*1000)
        #print("voltage : ",voltage)
        voltage[inhib != 0] = 0
        J[inhib != 0] = 0
        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )
        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = 0#-1 #Reset voltage
        voltage[refractory_time > 0] = 0#-1 #Refractory voltage
        refractory_time[spiked_mask] = self.tau_ref + t_spike
        # ----------------------------

        n += (dt / self.tau_n) * (self.inc_n * output - n)

        #AdaptiveLIF.step(self, dt, J, output, voltage, refractory_time, adaptation)
        inhib[inhib != 0] += - 1
        #J[...] = 0
        #output[...] = 0
        

#---------------------------------------------------------------------
#add builder for STDPLIF
#---------------------------------------------------------------------

@Builder.register(STDPLIF)
def build_STDPLIF(model, STDPlif, neurons):
    
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['pre_filtered'] = Signal(
        np.zeros(neurons.size_in), name="%s.pre_filtered" % neurons)
    model.sig[neurons]['post_filtered'] = Signal(
        np.zeros(neurons.size_in), name="%s.post_filtered" % neurons)
    model.sig[neurons]['inhib'] = Signal(
        np.zeros(neurons.size_in), name="%s.inhib" % neurons)
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in),name= "%s.adaptation" % neurons
    )
    # set neuron output for a given input
    model.add_op(SimNeurons(neurons=STDPlif,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            state={"voltage": model.sig[neurons]['voltage'],
                                    "refractory_time": model.sig[neurons]['refractory_time'],
                                    "adaptation": model.sig[neurons]['adaptation'],
                                    "inhib": model.sig[neurons]['inhib']
                                     }))









class CustomRule_post_v5(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.45,vthn=0.45):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        
        self.history = [0]

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            vmem = np.clip(self.signal_vmem_pre, -2, 1.5)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(vmem, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            self.w = np.clip((self.w + dt*(fun_post((self.w,vmem, self.vprog, self.vthp,self.vthn),*popt_tio2))*post_out_matrix*self.lr), 0, 1)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w
            
            return np.dot(self.w, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal



import os
import re
# import cv2


def gen_video(directory, f_prename):
    
    assert os.path.exists(directory)

    img_array = []
    for filename in os.listdir(directory):
        if f_prename in filename:
            nb = re.findall(r"(\d+).png", filename)
            if len(nb) == 1:
                img = cv2.imread(os.path.join(directory, filename))
                img_array.append((int(nb[0]), img))

    height, width, layers = img.shape
    size = (width, height)

    img_array = sorted(img_array, key=lambda x: x[0])
    video_path = os.path.join(directory, f"{f_prename}.avi")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), 2, size)

    for _, img in img_array:
        out.write(img)
    out.release()

    print(f"{video_path} generated successfully.")