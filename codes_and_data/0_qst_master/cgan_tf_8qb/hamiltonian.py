"""."""
import numpy as np
import qutip as qt
from scipy.constants import hbar

# Local paths:
local_path = "0_qst_master/cgan_tf_8qb/%s"
data_path = "0_qst_master/cgan_tf_8qb/data/%s"

# Qubit-qubit coupling matrix:

alpha = 1.47
constant = 67.3765

def create_matrix(rows, columns):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(columns):
            if i == j:
                entry = 0
                row.append(entry)
            else:
                entry = constant*np.power(abs(i - j), -alpha)
                row.append(entry)
        matrix.append(row)
    return matrix

num_qubits = 8

J_ij = create_matrix(num_qubits, num_qubits)
J_ij = np.matrix(J_ij)

# Magnetic field:

B_i = np.array([[2.85884931, 26.22601547, 48.5376065, 62.948092,
               53.7557919, 37.31269595, 10.50862743, 0.1]])

B = 15707.96326795

# Hamiltonian:

# Define the spin operators for each ion
spin_operators = [qt.jmat(0.5, 'x'), qt.jmat(0.5, 'y'), qt.jmat(0.5, 'z')]

# Define the raising and lowering operators for each ion
#s_plus = [qt.tensor([qt.qeye(2)] * i + [spin_operators[1]] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]
#s_minus = [qt.tensor([qt.qeye(2)] * i + [spin_operators[1]] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]
#s_z = [qt.tensor([qt.qeye(2)] * i + [spin_operators[2]] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]
#s_plus = [qt.tensor([qt.qeye(2)] * i + [spin_operators[1]] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]
#s_minus = [qt.tensor([qt.qeye(2)] * i + [spin_operators[2]] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]
#s_z = [qt.tensor([qt.qeye(2)] * i + [spin_operators[0]] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]
s_plus = [qt.tensor([qt.qeye(2)] * i + [qt.create(2)] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]
s_minus = [qt.tensor([qt.qeye(2)] * i + [qt.destroy(2)] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]
s_z = [qt.tensor([qt.qeye(2)] * i + [spin_operators[0]] + [qt.qeye(2)] * (num_qubits - i - 1)) for i in range(num_qubits)]

# XY interaction term
#interaction_term = sum([s_plus[i] * s_minus[i+1] + s_minus[i] * s_plus[i+1] for i in range(num_qubits-1)])
interaction_term = sum([J_ij[i, j] * (s_plus[i] * s_minus[j] + s_minus[i] * s_plus[j]) for i in range(num_qubits-1) for j in range(i+1, num_qubits)])

# Transverse magnetic field term
#transverse_field_term = hbar*B * sum(s_z)
transverse_field_term = B*sum(s_z)

# Define the Hamiltonian
#H = hbar*interaction_term + transverse_field_term
H = interaction_term + transverse_field_term

# Print the Hamiltonian
print("Hamiltonian:\n", H)

H = H.full()
#np.savetxt(data_path % "hamiltonian.txt", H, fmt="%s")
np.savetxt(data_path % 'hamiltonian.txt', H.view(float))