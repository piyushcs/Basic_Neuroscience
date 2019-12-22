# hebbian rule to calculate w
from numpy import linalg as LA
import numpy as np
import math

# match the principal eigen vector direction to check about the correct w

# def get_weight(evals, evecs, t):
# 	component_1 = np.array(math.exp(evals[0]*t) * evecs[:, 0])
# 	print(component_1)
# 	component_2 = np.array(math.exp(evals[1]*t) * evecs[:, 1])
# 	print(component_2)
# 	return component_1 + component_2

q = np.array([[0.15, 0.1], [0.1, 0.12]])

# q = np.array([[0.2, 0.1], [0.1, 0.3]])

# q = np.array([[0.2, 0.1], [0.1, 0.15]])

evals, evecs = LA.eig(q)

def get_angle(val):
	deg = math.degrees(math.atan(val))
	print(deg)

def get_length(v):
	mod = 0.0
	for element in v:
		mod += element * element
	mod = math.sqrt(mod)
	print(mod)

print(evecs, evals)

print("Principal eigen vector details")
get_angle(evecs[1, 0]/evecs[0, 0])
print(evecs[1, 0]/evecs[0, 0])
# get_length(evecs[:, 0])

results = [[-1.57, -1.23], [-1.51, -1.30], [0.89, 1.78], [1.05, 1.70]]

print("Given vectors for W")
for r in results:
	get_angle(r[1]/r[0])
	# print(r[1]/r[0])
	# get_length(r)

# evecs = np.array([[-0.85065, 0.52573], [-0.52573, -0.85065]])
# delta_t = 0.01

# w_t = get_weight(evals, evecs, delta_t)
# w_t_1 = get_weight(evals, evecs, 2*delta_t)

# delta_w = w_t_1 - w_t

# w = w_t
# print(w_t)
# for i in range(10):
# 	w = w + delta_w

# print(w)