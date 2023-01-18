import nengo
import nengo_loihi
import numpy as np

SEED = 45

def get_A_p_and_B_p_matrices(d):
  np.random.seed(SEED)
  A_p = 2 + np.random.rand(d, d)
  np.random.seed(SEED)
  B_p = 2 + np.random.rand(d, 1)

  return A_p, B_p

def get_enc_transform_matrix(d):
  t_mat  = np.zeros((d, 2*d))
  for i in range(0, d):
    t_mat[i, 2*i] = 1 # To make a positive copy of the scalar, since no Encoders.
    t_mat[i, 2*i+1] = -1 # To make a negative copy of the scalar, since no Encoders.

  return t_mat

def get_nengo_ldsnn_model(x, A_p, B_p, t_mat, lyr_e2h, lyr_h2o, d):
  print("Running the LDSNN network...")

  with nengo.Network() as net:
    inp = nengo.Node(output = lambda t: x[int(t*1000)-1])
    ldn = nengo.Node(size_in=d)
    enc = nengo.Ensemble(2*d, 1, neuron_type=nengo.SpikingRectifiedLinear())
    hdn = nengo.Ensemble(3*d, 1, neuron_type=nengo.SpikingRectifiedLinear())
    otp = nengo.Node(size_in=2)

    nengo.Connection(inp, ldn, transform=B_p)
    nengo.Connection(ldn, ldn, transform=A_p)

    nengo.Connection(ldn, enc.neurons, transform=t_mat.T)
    nengo.Connection(enc.neurons, hdn.neurons, transform=lyr_e2h.T)
    nengo.Connection(hdn.neurons, otp, transform=lyr_h2o.T)

    net.p = nengo.Probe(otp)

  return net

if __name__ == "__main__":
  T = 140
  d = 10
  np.random.seed(SEED)
  x = np.random.rand(T) - 1.0
  print(x, x.shape)

  A_p, B_p = get_A_p_and_B_p_matrices(d)
  t_mat = get_enc_transform_matrix(d)
  np.random.seed(SEED)
  lyr_e2h = np.random.rand(2*d, 3*d)
  np.random.seed(SEED)
  lyr_h2o = np.random.rand(3*d, 2)

  net = get_nengo_ldsnn_model(x, A_p, B_p, t_mat, lyr_e2h, lyr_h2o, d)

  with nengo_loihi.Simulator(net) as sim:
    sim.run(T/1000)

  print(sim.data[net.p], sim.data[net.p].shape)
