from __future__ import print_function
"""
Created on Wed Apr 22 16:02:53 2015

Basic integrate-and-fire neuron 
R Rao 2007

translated to Python by rkp 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# input current
I = 8.051 # nA
# I = .251 # 250 pA
# capacitance and leak resistance
C = 1 # nF
R = 40 # M ohms

# I & F implementation dV/dt = - V/RC + I/C
# Using h = 1 ms step size, Euler method

V = 0
tstop = 200
abs_ref = 5 # absolute refractory period 
ref = 0 # absolute refractory period counter
V_trace = []  # voltage trace for plotting
V_th = 10 # spike threshold
Spikes = 0 # spike count

for t in range(tstop):
	# print(V)

	if not ref:
		V = V - (V/(R*C)) + (I/C)
	else:
		ref -= 1
		V = 0.2 * V_th # reset voltage

	if V > V_th:
		Spikes += 1
		V = 50 # emit spike
		ref = abs_ref # set refractory counter

	V_trace += [V]

	# time.sleep(0.2)

plt.plot(V_trace)
plt.show()

print("No of spikes:", Spikes)
print("Maximum firing rate:", Spikes*1000/tstop)