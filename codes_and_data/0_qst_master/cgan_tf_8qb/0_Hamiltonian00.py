import numpy as np
from scipy.sparse import kron, eye, csc_matrix
from scipy.linalg import expm

# Local paths:
local_path = "0_qst_master/cgan_tf_8qb/%s"
data_path = "0_qst_master/cgan_tf_8qb/data/%s"

#Number of qubits
M = 8

Jij = np.array([[0., 54.17798091, 17.8394829, 7.84929745, 3.90269405,
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

Jij = np.matrix(Jij)

# Magnetic field:
Bi = np.array([4.56473413, 8.83730013, 11.26260966, 12.92765377,
               10.75367165, 9.33681337, 0., 1.57079633])

B = 9424.77796077

# Pauli matrices
sx = csc_matrix([[0, 1], [1, 0]])
sz = csc_matrix([[1, 0], [0, -1]])
id = eye(2)

# operators acting on spins
sxns = []
szns = []
for iM in range(1, M+1):
    sxns.append(kron(kron(eye(2**(M-iM)), sx), eye(2**(iM-1))))
    szns.append(kron(kron(eye(2**(M-iM)), sz), eye(2**(iM-1))))

# Hamiltonian
H = csc_matrix((2**M, 2**M))
for i1 in range(1, M+1):
    for i2 in range(1, M+1):
        if i1 != i2:
            H += Jij[i1-1, i2-1] * sxns[i1-1] * sxns[i2-1]
    H += (B + Bi[i1-1]) * szns[i1-1]

H = H.toarray()

#np.savetxt(data_path % 'hamiltonian.txt', H.view(float))
np.savetxt(data_path % 'hamiltonian.txt', H)

timeEvol = expm(-1j * dt * H)  # or use the additional function expv instead of expm for large M