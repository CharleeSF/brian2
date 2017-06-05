#!/usr/bin/env python
'''
Input-Frequency curve of a IF model.
Network: 1000 unconnected integrate-and-fire neurons (leaky IF)
with an input parameter v0.
The input is set differently for each neuron.
'''
from brian2 import *
from shutil import copyfile

prefs.codegen.target = 'cython'

prefs.codegen.cpp.libraries += ['gsl', 'gslcblas']
prefs.codegen.cpp.headers += ['gsl/gsl_odeiv2.h']
prefs.codegen.cpp.include_dirs += ['/home/charlee/softwarefolder/gsl-2.3/gsl/']

tau = 10*ms
eqs = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''

def get_time(l, description):
    for name, time in l:
        if description in name:
            return time


def run_net(N, version, duration=1*second):

    copyfile('cythoncode_%s.pyx'%version, 'cythoncode_tempfile.pyx')
    group = NeuronGroup(N, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method='euler')
    group.v = -10*mV
    group.v0 = '20*mV * i / (N-1)'

    monitor = SpikeMonitor(group)

    net = Network(group, monitor)
    net.run(duration)
    #plot(group.v0/mV, monitor.count / duration)
    #show()
    return get_time(net.profiling_info, 'stateupdater')

for N in [10,100,1000]:
    print run_net(N, 'equations')

#plot(group.v0/mV, monitor.count / duration)


