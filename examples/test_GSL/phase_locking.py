#!/usr/bin/env python
'''
Phase locking of IF neurons to a periodic input.

23/06/2017 Edit: adapted to integrate with GSL code (output is the same as with normal cython code)
'''
from brian2 import *

prefs.codegen.target = 'cython'

tau = 20*ms
n = 100
b = 1.2 # constant current mean, the modulation varies
freq = 10*Hz

eqs = '''
dv/dt = (-v + a * sin(2 * pi * freq * t) + b) / tau : 1
a : 1
'''
neurons = NeuronGroup(n, model=eqs, threshold='v > 1', reset='v = 0',
                      method='GSL_stateupdater')
neurons.state_updater.codeobj_class = GSLCythonCodeObject
neurons.v = 'rand()'
neurons.a = '0.05 + 0.7*i/n'
S = SpikeMonitor(neurons)
trace = StateMonitor(neurons, 'v', record=50)

run(1000*ms)

print neurons.state_updater.codeobj.code

subplot(211)
plot(S.t/ms, S.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
subplot(212)
plot(trace.t/ms, trace.v.T)
xlabel('Time (ms)')
ylabel('v')
show()