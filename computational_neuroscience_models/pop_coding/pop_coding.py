from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import pickle
import time, math

FILENAME = 'tuning_3.4.pickle'

with open(FILENAME, 'rb') as f:
    data = pickle.load(f)

# question 7

tuning_curves = {}
for neuron in ['neuron1', 'neuron2', 'neuron3', 'neuron4']:
    # axis=0 means 'average columns'
    tuning_curves[neuron] = np.mean(data[neuron], axis=0)
    plt.plot(data['stim'], tuning_curves[neuron])
    plt.xlabel('Direction (degrees)')
    plt.ylabel('Mean firing rate')
    plt.title('Tuning curve for ' + neuron)
    plt.show()


# question 8

T = 10 # in seconds
for neuron in ['neuron1', 'neuron2', 'neuron3', 'neuron4']:
    mean_spikes = T * np.mean(data[neuron], axis=0)
    spikes_variance = (T**2) * np.var(data[neuron], axis=0)
    plt.scatter(mean_spikes, spikes_variance)
    plt.xlabel('Mean number of spikes')
    plt.ylabel('Variance of number of spikes')
    plt.title('Fano test for ' + neuron)
    plt.show()

# question 9

with open('pop_coding_3.4.pickle', 'rb') as f:
    data2 = pickle.load(f)

v = np.array([0,0], dtype='float64')

for a in ['1', '2', '3', '4']:
    v += (np.average(data2['r'+a]) /
               max(tuning_curves['neuron'+a])) * data2['c'+a]

angle = np.degrees(np.arctan(v[1]/v[0])) #anticlockwise

angle = -angle #clockwise

angle = int(angle + 90) #shift axes from x-positive to y-positive

print(angle, "Degrees")
