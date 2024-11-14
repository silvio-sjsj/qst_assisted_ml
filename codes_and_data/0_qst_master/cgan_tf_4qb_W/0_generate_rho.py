"""Generate density matrix for the 4-butis W state."""
import numpy as np
import scipy as sp
import scipy.linalg
import numpy.random

# Local paths:
local_path = "0_qst_master/cgan_tf_4qb_W/%s"
data_path = "0_qst_master/cgan_tf_4qb_W/data/%s"

zero = np.array([[1.0],
                 [0.0]])

one = np.array([[0.0],
                [1.0]])


def NormalizeState(state):
    """Normalize a given state."""
    return state/sp.linalg.norm(state)


def NKron(*args):
    """Calculate a Kronecker product over a variable number of inputs."""
    result = np.array([[1.0]])
    for op in args:
        result = np.kron(result, op)
    return result


w_a = NKron(zero, zero, zero, one)
w_b = NKron(zero, zero, one, zero)
w_c = NKron(zero, one, zero, zero)
w_d = NKron(one, zero, zero, zero)

psi_w_state = NormalizeState(w_a + w_b + w_c + w_d)

rho_w_state = np.dot(psi_w_state, psi_w_state.T)

np.savetxt(data_path % "psi_4qubits_W.txt", psi_w_state, fmt="%s")
np.savetxt(data_path % "rho_4qubits_W.txt", rho_w_state, fmt="%s")