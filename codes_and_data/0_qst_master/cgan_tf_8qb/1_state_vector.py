"""."""
import numpy as np
import scipy as sp
import qutip as qt
import math
from scipy.linalg import expm
from qutip import Qobj

# Local paths:
local_path = "0_qst_master/cgan_tf_8qb/%s"
data_path = "0_qst_master/cgan_tf_8qb/data/%s"

# Nèel-ordered state

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

psi_neel = NKron(one, zero, one, zero, one, zero, one, zero)
psi_neel = NormalizeState(psi_neel)

rho_neel = np.dot(psi_neel, psi_neel.T)

time_vec = [0, 0.0005, 0.001 , 0.0015, 0.002 , 0.0025, 0.003 , 0.0035]
time_vec2 = [0.00075, 0.0015, 0.00225, 0.003, 0.00375, 0.0045, 0.00525, 0.006]

#H = np.loadtxt(data_path % 'hamiltonian.txt').view(complex)
#H = np.loadtxt(data_path % 'hamiltonian_ch.txt')
H = np.loadtxt(data_path % 'hamiltonian.txt')
   
for t in time_vec:
    psi_neel_t = np.matmul(expm(-1j*H*t), psi_neel)
    psi_neel_t = NormalizeState(psi_neel_t)
    rho_neel_t = np.dot(psi_neel_t, psi_neel_t.conjugate().T)  # Calculate Hermitian conjugate
    #rho_neel_t = (rho_neel_t + rho_neel_t.conjugate().T) / 2  # Ensure Hermitian property
    rho_neel_tt = Qobj(rho_neel_t)
    print(rho_neel_tt)
    print(np.trace(rho_neel_t))
    np.savetxt(data_path % "rho_neel_{p}.txt".format(p=t), rho_neel_t.view(complex))

for t in time_vec2:
    psi_neel_t = np.matmul(expm(-1j*H*t), psi_neel)
    psi_neel_t = NormalizeState(psi_neel_t)
    rho_neel_t = np.dot(psi_neel_t, psi_neel_t.conjugate().T)  # Calculate Hermitian conjugate
    #rho_neel_t = (rho_neel_t + rho_neel_t.conjugate().T) / 2  # Ensure Hermitian property
    rho_neel_tt = Qobj(rho_neel_t)
    print(rho_neel_tt)
    print(np.trace(rho_neel_t))
    np.savetxt(data_path % "rho_neelbcm_{p}.txt".format(p=t), rho_neel_t.view(complex))