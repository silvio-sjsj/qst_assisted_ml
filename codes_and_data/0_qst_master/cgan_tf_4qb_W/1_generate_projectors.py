"""."""
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
from itertools import product
from qutip import basis, Qobj, dag, tensor
from jax import jit

# Local paths:
local_path = "0_qst_master/cgan_tf_4qb_W/%s"
data_path = "0_qst_master/cgan_tf_4qb_W/data/%s"

n = 4

# Carregar estado
estado = np.loadtxt(data_path % "rho_4qubits_W.txt")

rho = Qobj(estado[:])

# Projetores p/ 1-qubit

a = basis(2, 0)*basis(2, 0).dag()
b = basis(2, 1)*basis(2, 1).dag()
c = (0.5)*(basis(2, 1) + basis(2, 0))*(basis(2, 1) + basis(2, 0)).dag()
d = (0.5)*(-basis(2, 1) + basis(2, 0))*(-basis(2, 1) + basis(2, 0)).dag()
e = (0.5)*(1j*basis(2, 1) + basis(2, 0))*(1j*basis(2, 1) + basis(2, 0)).dag()
f = (0.5)*(-1j*basis(2, 1) + basis(2, 0))*(-1j*basis(2, 1) + basis(2, 0)).dag()

projetores = [a, b, c, d, e, f]

m_ops = [] # measurement ops

for idx in product([0, 1, 2, 3, 4, 5], repeat=n):
    U = tensor([projetores[i] for i in idx])
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
data = jnp.array(expvals + np.random.uniform(0, 0.05, size=expvals.shape[0]))

labels = np.arange(0, len(data))

plt.bar(labels, data, label="Noisy")
plt.bar(labels, expvals, label="True", width=0.5)
plt.legend()
plt.xlabel("Measurement setting")
plt.ylabel("Expectation value")

plt.show()