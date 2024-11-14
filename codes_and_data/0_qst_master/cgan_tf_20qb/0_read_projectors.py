"""."""
import numpy as np
import tensorflow as tf
from qutip import tensor
from qutip import sigmax, sigmaz, sigmay
from tqdm.auto import tqdm

# Local paths:
local_path = "0_qst_master/cgan_tf_20qb/%s"
data_path = "0_qst_master/cgan_tf_20qb/data/%s"

# Reading projectors
projs_settings = np.loadtxt(data_path % 'measurement_settings.txt', dtype=str)

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
    m_ops.append(U)

###################################################
jaxx = scipy.sparse.csr_matrix(m_ops[0].data)

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

jj = convert_sparse_matrix_to_sparse_tensor(jaxx)

###############################################################

import numpy as np
from qutip import Qobj

def convert_to_batches(qutip_obj, batch_size):
    num_elements = qutip_obj.shape[0]
    num_batches = int(np.ceil(num_elements / batch_size))

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_elements)
        batch = qutip_obj[start_idx:end_idx]
        yield batch

# Example usage
large_qutip_obj = Qobj(...)  # Replace ... with your large QuTiP object
batch_size = 1000

# Process batches incrementally
for batch in convert_to_batches(large_qutip_obj, batch_size):
    numpy_array = np.array(batch)
    # Perform necessary operations on the numpy_array
    # ...
    # Discard the numpy_array to free memory
    del numpy_array