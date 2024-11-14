"""."""
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
from itertools import product
from qutip import basis, Qobj, dag, tensor
from jax import jit
from scipy.io import loadmat

# Local paths:
local_path = "0_qst_master/cgan_tf_20qb/%s"
data_path = "0_qst_master/cgan_tf_20qb/data/%s"

data = loadmat(data_path % 'data_20.mat')
data = data['data']

x_0 = data[0][0]
x_1 = data[0][1]
x_2 = data[0][2]
x_3 = data[0][3]
x_4 = data[0][4]
x_5 = data[0][5]
x_6 = data[0][6]
x_7 = data[0][7]