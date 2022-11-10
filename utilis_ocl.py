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

from nengo_extras.graphviz import net_diagram
import nengo_ocl

from nengo.neurons import LIFRate

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
#from nengo_extras.neurons import spikes2events
from nengo.utils.numpy import clip
import numpy as np
import random
import math

import nengo.dists as nengod
import numpy as np
import pyopencl as cl
from mako.template import Template
from nengo.utils.numpy import is_number

from nengo_ocl import ast_conversion
from nengo_ocl.clraggedarray import CLRaggedArray, to_device
from nengo_ocl.plan import Plan
from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.utils import as_ascii, indent, nonelist, round_up




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
            V = -1;
            spiked = 1;
        }
% if not fastlif:
         else if (V < 0) {
            V = V;
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


class CustomOCLSimulator(nengo_ocl.Simulator):
   def _plan_MyLIF_in(self, ops):
        if not all(op.neurons.min_voltage == 0 for op in ops):
            raise NotImplementedError("LIF min voltage")
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        V = self.all_data[[self.sidx[op.state["voltage"]] for op in ops]]
        W = self.all_data[[self.sidx[op.state["refractory_time"]] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray(
            [op.neurons.tau_ref * np.ones(op.J.size) for op in ops], dtype=J.dtype
        )
        tau = self.RaggedArray(
            [op.neurons.tau_rc * np.ones(op.J.size) for op in ops], dtype=J.dtype
        )
        amp = self.RaggedArray(
            [op.neurons.amplitude * np.ones(op.J.size) for op in ops], dtype=J.dtype
        )
        return [plan_MyLIF_in(self.queue, dt, J, V, W, S, ref, tau, amp)]




def _plan_template(  # noqa: C901
    queue,
    name,
    core_text,
    declares="",
    tag=None,
    blockify=True,
    inputs=None,
    outputs=None,
    parameters=None,
):
    """Template for making a plan for vector nonlinearities.
    This template assumes that all inputs and outputs are vectors.
    Parameters
    ----------
    blockify : bool
        If true, divide the inputs up into blocks with a maximum size.
    inputs: dictionary of CLRaggedArrays
        Inputs to the function. RaggedArrays must be a list of vectors.
    outputs: dictionary of CLRaggedArrays
        Outputs of the function. RaggedArrays must be a list of vectors.
    parameters: dictionary of CLRaggedArrays
        Parameters to the function. Each RaggedArray element must be a vector
        of the same length of the inputs, or a scalar (to be broadcasted).
        Providing a float instead of a RaggedArray makes that parameter
        constant.
    """
    inputs = {} if inputs is None else inputs
    outputs = {} if outputs is None else outputs
    parameters = {} if parameters is None else parameters

    input0 = list(inputs.values())[0]  # input to use as reference for lengths

    # split parameters into static and updated params
    static_params = {}  # static params (hard-coded)
    params = {}  # variable params (updated)
    for k, v in parameters.items():
        if isinstance(v, CLRaggedArray):
            params[k] = v
        elif is_number(v):
            static_params[k] = ("float", float(v))
        else:
            raise ValueError(
                "Parameter %r must be CLRaggedArray or float (got %s)" % (k, type(v))
            )

    avars = {}
    bw_per_call = 0
    for vname, v in list(inputs.items()) + list(outputs.items()) + list(params.items()):
        assert vname not in avars, "Name clash"
        assert len(v) == len(input0)
        assert (v.shape0s == input0.shape0s).all()
        assert (v.stride0s == v.shape1s).all()  # rows contiguous
        assert (v.stride1s == 1).all()  # columns contiguous
        assert (v.shape1s == 1).all()  # vectors only

        offset = "%(name)s_starts[gind1]" % {"name": vname}
        avars[vname] = (v.ctype, offset)
        bw_per_call += v.nbytes

    ivars = {k: avars[k] for k in inputs}
    ovars = {k: avars[k] for k in outputs}
    pvars = {k: avars[k] for k in params}

    fn_name = str(name)
    textconf = dict(
        fn_name=fn_name,
        declares=declares,
        core_text=core_text,
        ivars=ivars,
        ovars=ovars,
        pvars=pvars,
        static_params=static_params,
    )

    text = """
    ////////// MAIN FUNCTION //////////
    __kernel void ${fn_name}(
% for name, [type, offset] in ivars.items():
        __global const int *${name}_starts,
        __global const ${type} *${name}_buf,
% endfor
% for name, [type, offset] in ovars.items():
        __global const int *${name}_starts,
        __global ${type} *${name}_buf,
% endfor
% for name, [type, offset] in pvars.items():
        __global const int *${name}_starts,
        __global const int *${name}_shape0s,
        __global const ${type} *${name}_buf,
% endfor
        __global const int *sizes
    )
    {
        const int gind0 = get_global_id(0);
        const int gind1 = get_global_id(1);
        if (gind1 >= ${N} || gind0 >= sizes[gind1])
            return;
% for name, [type, offset] in ivars.items():
        ${type} ${name} = ${name}_buf[${offset} + gind0];
% endfor
% for name, [type, offset] in ovars.items():
        ${type} ${name};
% endfor
% for name, [type, offset] in pvars.items():
        const ${type} ${name} = ${name}_buf[${offset} + gind0];
% endfor
% for name, [type, value] in static_params.items():
        const ${type} ${name} = ${value};
% endfor
        //////////////////////////////////////////////////
        //vvvvv USER DECLARATIONS BELOW vvvvv
        ${declares}
        //^^^^^ USER DECLARATIONS ABOVE ^^^^^
        //////////////////////////////////////////////////
        /////vvvvv USER COMPUTATIONS BELOW vvvvv
        ${core_text}
        /////^^^^^ USER COMPUTATIONS ABOVE ^^^^^
% for name, [type, offset] in ovars.items():
        ${name}_buf[${offset} + gind0] = ${name};
% endfor
    }
    """

    if blockify:
        # blockify to help with heterogeneous sizes

        # find best block size
        block_sizes = [16, 32, 64, 128, 256, 512, 1024]
        N = np.inf
        for block_size_i in block_sizes:
            sizes_i, inds_i, _ = blockify_vector(block_size_i, input0)
            if len(sizes_i) < N:
                N = len(sizes_i)
                block_size = block_size_i
                sizes = sizes_i
                inds = inds_i

        clsizes = to_device(queue, sizes)
        get_starts = lambda ras: [
            to_device(queue, starts) for starts in blockify_vectors(block_size, ras)[2]
        ]
        Istarts = get_starts(inputs.values())
        Ostarts = get_starts(outputs.values())
        Pstarts = get_starts(params.values())
        Pshape0s = [to_device(queue, x.shape0s[inds]) for x in params.values()]

        lsize = None
        gsize = (block_size, len(sizes))

        full_args = []
        for vstarts, v in zip(Istarts, inputs.values()):
            full_args.extend([vstarts, v.cl_buf])
        for vstarts, v in zip(Ostarts, outputs.values()):
            full_args.extend([vstarts, v.cl_buf])
        for vstarts, vshape0s, v in zip(Pstarts, Pshape0s, params.values()):
            full_args.extend([vstarts, vshape0s, v.cl_buf])
        full_args.append(clsizes)
    else:
        # Allocate more than enough kernels in a matrix
        lsize = None
        gsize = (input0.shape0s.max(), len(input0))

        full_args = []
        for v in inputs.values():
            full_args.extend([v.cl_starts, v.cl_buf])
        for v in outputs.values():
            full_args.extend([v.cl_starts, v.cl_buf])
        for vname, v in params.items():
            full_args.extend([v.cl_starts, v.cl_shape0s, v.cl_buf])
        full_args.append(input0.cl_shape0s)

    textconf["N"] = gsize[1]
    text = as_ascii(Template(text, output_encoding="ascii").render(**textconf))
    fns = cl.Program(queue.context, text).build()
    _fn = getattr(fns, fn_name)
    _fn.set_args(*[arr.data for arr in full_args])

    plan = Plan(queue, _fn, gsize, lsize=lsize, name=name, tag=tag)
    plan.full_args = tuple(full_args)  # prevent garbage-collection
    plan.bw_per_call = bw_per_call
    plan.description = "groups: %d; items: %d; items/group: %0.1f [%d, %d]" % (
        gsize[1],
        input0.sizes.sum(),
        input0.sizes.mean(),
        input0.sizes.min(),
        input0.sizes.max(),
    )
    return plan


def blockify_vectors(max_size, ras):
    ras = list(ras)
    ra0 = ras[0] if len(ras) > 0 else None
    N = len(ra0) if ra0 is not None else 0
    for ra in ras:
        assert len(ra) == N
        assert (ra.shape1s == 1).all()
        assert (ra.shape0s == ra0.shape0s).all()

    sizes = []
    inds = []
    starts = [[] for _ in ras]
    for i in range(N):
        size = ra0.shape0s[i]
        startsi = [ra.starts[i] for ra in ras]
        while size > 0:
            sizes.append(min(size, max_size))
            size -= max_size
            inds.append(i)
            for k, ra in enumerate(ras):
                starts[k].append(startsi[k])
                startsi[k] += max_size * ra.stride0s[i]

    return (
        np.array(sizes, dtype=np.int32),
        np.array(inds, dtype=np.int32),
        np.array(starts, dtype=np.int32),
    )


def blockify_vector(max_size, ra):
    sizes, inds, starts = blockify_vectors(max_size, [ra])
    return sizes, inds, starts[0]





