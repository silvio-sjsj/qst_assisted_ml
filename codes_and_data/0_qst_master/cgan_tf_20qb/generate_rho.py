"""Generate density matrix for the 4-butis GHZ state."""
import numpy as np
import scipy as sp
import scipy.linalg
import numpy.random

# Local paths:
local_path = "0_qst_master/cgan_tf_20qb/%s"
data_path = "0_qst_master/cgan_tf_20qb/data/%s"

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


four_qubitstate_zero = NKron(zero, zero, zero, zero)
four_qubitstate_one = NKron(one, one, one, one)

psi_ghz_state = NormalizeState(four_qubitstate_zero + four_qubitstate_one)

rho_ghz_state = np.dot(psi_ghz_state, psi_ghz_state.T)

np.savetxt(data_path % "psi_4qubits_GHZ.txt", psi_ghz_state,fmt="%s")
np.savetxt(data_path % "rho_4qubits_GHZ.txt", rho_ghz_state,fmt="%s")

# Global depolarization probability:

for j in range(10):
    p_dep = np.random.uniform(0, 0.5)
    rho_depolarized = (1 - p_dep)*rho_ghz_state + ((p_dep)/16)*np.eye(16)
    np.savetxt(
        data_path % "rho_4qubits_GHZ_dep_{p}.txt".format(p=p_dep),
        rho_depolarized,fmt="%s")