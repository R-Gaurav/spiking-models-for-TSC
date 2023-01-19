import numpy as np
import nengo

from pyJoules.energy_meter import measure_energy
from lsnn import (get_A_p_and_B_p_matrices, get_enc_transform_matrix,
                  get_nengo_lsnn_model)
from slrc import get_nengo_slrc_model

SEED = 45

@measure_energy
def execute_spiking_model_on_cpu(T):
    with nengo.Simulator(net) as sim:
        sim.run(T/1000)

if __name__=="__main__":
    d = 10
    T = 140
    np.random.seed(SEED)
    x = np.random.rand(T) - 1.0
    print(x, x.shape)

    A_p, B_p = get_A_p_and_B_p_matrices(d)
    t_mat = get_enc_transform_matrix(d)
    np.random.seed(SEED)
    lyr_e2h = np.random.rand(2*d, 3*d)
    np.random.seed(SEED)
    lyr_h2o = np.random.rand(3*d, 2)
    
    # Comment one of the below to run either SLRC or LSNN.
    net = get_nengo_lsnn_model(x, A_p, B_p, t_mat, lyr_e2h, lyr_h2o, d)
    #net = get_nengo_slrc_model(x, A_p, B_p, d, n_nrns=100)

    execute_spiking_model_on_cpu(T)

