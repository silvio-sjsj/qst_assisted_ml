"""."""
import numpy as np
import qutip as qt
from scipy.constants import hbar

# Local paths:
local_path = "0_qst_master/cgan_tf_8qb/%s"
data_path = "0_qst_master/cgan_tf_8qb/data/%s"

# Qubit-qubit coupling matrix:

num_qubits = 8

J_ij = np.array([[0., 54.17798091, 17.8394829, 7.84929745, 3.90269405,
            2.08339415, 1.14951297, 0.60136459],
                 [54.17798091, 0., 70.59885055, 24.835975, 10.59639265,
            4.9922523, 2.4617254, 1.15291741],
                 [17.8394829, 70.59885055, 0., 76.86678958, 27.21106841,
            11.23662521, 4.94498838, 2.06978154],
                 [7.84929745, 24.835975, 76.86678958, 0., 78.56276494,
            27.16047657, 10.47655709, 3.86998576],
                 [3.90269405, 10.59639265, 27.21106841, 78.56276494, 0.,
            75.66091576, 24.21490747, 7.67567727],
                 [2.08339415, 4.9922523, 11.23662521, 27.16047657, 75.66091576,
            0., 68.70542363, 17.41245362],
                 [1.14951297, 2.4617254, 4.94498838, 10.47655709, 24.21490747,
            68.70542363, 0., 52.38045613],
                 [0.60136459, 1.15291741, 2.06978154, 3.86998576, 7.67567727,
            17.41245362, 52.38045613, 0.]])

J_ij = np.matrix(J_ij)

# Magnetic field:

B_i = np.array([[4.56473413, 8.83730013, 11.26260966, 12.92765377,
                 10.75367165, 9.33681337, 0., 1.57079633]])

B = np.array([[9424.77796077]])
B = 9424.77796077

# Hamiltonian:

# Define the spin operators for each ion
spin_operators = [qt.jmat(0.5, 'x'), qt.jmat(0.5, 'y'), qt.jmat(0.5, 'z')]

# Define the raising and lowering operators for each ion
s_plus = [qt.tensor([qt.qeye(2)] * i + [qt.create(2)] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]
s_minus = [qt.tensor([qt.qeye(2)] * i + [qt.destroy(2)] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]
s_z = [qt.tensor([qt.qeye(2)] * i + [spin_operators[0]] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]

# XY interaction term
interaction_term = sum([J_ij[i, j] * (s_plus[i] * s_minus[j] + s_minus[i] * s_plus[j]) for i in range(num_qubits-1) for j in range(i+1, num_qubits)])

# Transverse magnetic field term

# Reshape B_i to match the shape of s_z
B_i_reshaped = np.reshape(B_i, (num_qubits, 1))

# Transverse magnetic field term
transverse_field_term = sum((B_i_reshaped[i][0] + B) * s_z[i] for i in range(num_qubits))

# Define the Hamiltonian
H = interaction_term + transverse_field_term

# Print the Hamiltonian
print("Hamiltonian:\n", H)

H = H.full()
#np.savetxt(data_path % "hamiltonian.txt", H, fmt="%s")
np.savetxt(data_path % 'hamiltonian.txt', H.view(float))