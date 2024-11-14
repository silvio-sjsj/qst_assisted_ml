"""."""
import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor
from qutip import sigmax, sigmaz, sigmay
from tqdm.auto import tqdm
from jax import jit
from jax import numpy as jnp
from itertools import product
from qutip import basis, Qobj, dag, tensor

# Local paths:
local_path = "0_qst_master/cgan_tf_8qb/%s"
data_path = "0_qst_master/cgan_tf_8qb/data/%s"

# Number of qubits:
n = 8

# Carregar estado
estado = np.loadtxt(data_path % "rho_neel_0.txt", dtype=np.complex_)
#estado = np.loadtxt(data_path % "rho_neel_0.0035.txt", dtype=np.complex_)

rho = Qobj(estado[:])

# Reading projectors
projs_settings = np.loadtxt(data_path % 'measurement_settings2.txt', dtype=str)

X = sigmax()
Y = sigmay()
Z = sigmaz()

m_ops = [] # measurement operators

def string_to_operator(basis):  
    mat_real = []
    
    for j in range(len(basis)):
        if basis[j] == 'X':
            mat_real.append(X)     
        if basis[j] =='Y':
            mat_real.append(Y)     
        if basis[j] =='-Y':
            mat_real.append(-Y)     
        if basis[j] == 'Z':
            mat_real.append(Z)   
    return mat_real

for i in range(27):
    U = string_to_operator(projs_settings[i])
    U = tensor(U)
    #m_ops.append(U)
    m_ops.extend([U] * 20)

#########################################
@jit # just in time compilation for speed
def expectation(rho, ops):
    """Expectation layer that predicts the expectation value for operators

    Args:
        rho (array): A density matrix as a Jax array of shape (N, N)
        ops (array): A set of measurement operators as a Jax array of shape
                     (n, N, N) where n is the number of operators

    Returns
        evals (array): An array of expectation values for the operators.
    """
    products = jnp.einsum('nij, jk -> nik', ops, rho)
    trace = jnp.einsum('nii -> n', products)
    return trace.real

ops_jax = jnp.array(m_ops)
rho_jax = jnp.array(rho)

expvals = expectation(rho_jax, ops_jax)

# Sample/add noise to the expvals
data = jnp.array(expvals + np.random.uniform(0, 0.05, size=expvals.shape[0]))

labels = np.arange(0, len(data))

plt.bar(labels, data, label="Noisy")
plt.bar(labels, expvals, label="True", width=0.5)
plt.legend()
plt.xlabel("Measurement setting")
plt.ylabel("Expectation value")

plt.show()