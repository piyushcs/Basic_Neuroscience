import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import math

def calculate_w(X):
	w = np.random.rand(1, 2)
	delta_t = 0.01
	sigma = 1
	alpha = 1

	u = X[0]

	for i in range(1000):
		data_point_index = i % 100
		u = np.array(X[data_point_index]).reshape(1, 2)

		v = u[0, 0] * w[0, 0] + u[0, 1] * w[0 , 1]
		delta_w = delta_t * (v * u - (v * v) * w) # for oja rule
		# delta_w = delta_t * (v * u) # for hebb rule
		w = w + delta_w

	return w

def get_angle(val):
	deg = math.degrees(math.atan(val))
	print(deg)

def get_length(v):
	mod = 0.0
	for element in v:
		mod += element * element
	mod = math.sqrt(mod)
	print(mod)



with open('c10p1.pickle', 'rb') as f:
	data = pickle.load(f)

data = np.array(data['c10p1'])
mean_centered_data = np.array(data - data.mean(axis=0))
correlation_mat = np.dot(np.transpose(mean_centered_data), mean_centered_data)
# print(correlation_mat)

evals, evecs = LA.eig(correlation_mat)
print(evecs)

w = calculate_w(mean_centered_data) # change mean_centered_data to data for testing without mean centering
print(w)

print("angle")
get_angle(w[0][1]/w[0][0])
get_angle(evecs[1, 0]/evecs[0, 0])

print("length")
get_length(w[0])
get_length(evecs[:, 0])