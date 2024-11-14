"""."""
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
from itertools import product
from qutip import basis, Qobj, dag, tensor
from qutip import sigmax, sigmaz, sigmay
from jax import jit

# Local paths:
local_path = "0_qst_master/cgan_tf_4qb_W/%s"
data_path = "0_qst_master/cgan_tf_4qb_W/data/%s"

# Number of qubits
n = 4

# Carregar estado
estado = np.loadtxt(data_path % "rho_4qubits_W.txt")

rho = Qobj(estado[:])

# Local qubit unitaries
unitaries = [sigmax(), sigmay(), sigmaz()]

m_ops = [] # measurement ops

for idx in product([0, 1, 2], repeat=n):
    U = tensor([unitaries[i] for i in idx])
    #U = jnp.array(U.full())
    m_ops.append(U)

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
#data = jnp.array(expvals + np.random.normal(0, 0.05, size=expvals.shape[0]))
data = jnp.array(expvals + np.random.uniform(-0.05, 0.05, size=expvals.shape[0]))

fig, ax = plt.subplots()

labels = np.arange(0, len(data))

ax.bar(labels, data, label="Noisy")
ax.bar(labels, expvals, label="True", width=0.5)
ax.set_title('Expectation values for every measurement settings of simulation cGAN-1b - Noise from Normal distribution')
#ax.set_title('Expectation values for every measurement settings of simulation cGAN-1b - Noise from Uniform distribution')
plt.legend(fontsize='20')
plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
plt.xlabel("Measurement setting", fontsize='20')
plt.ylabel("Expectation value", fontsize='20')

plt.show()

###############################################
# USE ONLY IN THE CASE OF NO ADDITION OF NOISE:

# add an epsilon to expvals to force zero bins to be plotted:
data = expvals
epsilon = 0.01
data = [x+epsilon for x in data]
data = np.array(data)

fig, ax = plt.subplots()

labels = np.arange(0, len(data))

ax.bar(labels, data, label="True", color='darkorange')
ax.set_title('Expectation values for every measurement settings of simulation cGAN-w_1a - No addition of noise')
plt.legend(fontsize='20')
plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
plt.xlabel("Measurement setting", fontsize='20')
plt.ylabel("Expectation value", fontsize='20')

plt.show()