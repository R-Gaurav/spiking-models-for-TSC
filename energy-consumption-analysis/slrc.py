import nengo
import nengo_loihi
import numpy as np

SEED = 45

def get_A_p_and_B_p_matrices(d):
  np.random.seed(SEED)
  A_p = 2+np.random.rand(d, d)
  np.random.seed(SEED)
  B_p = 2+np.random.rand(d, 1)

  return A_p, B_p

def get_nengo_slrc_model(x, A_p, B_p, d, n_nrns):
  print("Running the SLRC network...")

  with nengo.Network() as net:
    inp = nengo.Node(output = lambda t: x[int(t*1000)-1])
    ldn = nengo.networks.EnsembleArray(
      n_neurons=n_nrns,
      n_ensembles=d,
      neuron_type=nengo.SpikingRectifiedLinear(),
      max_rates=nengo.dists.Uniform(80, 120),
      intercepts=nengo.dists.Uniform(-1, 0.5),
      radius=1.5
    )
    ens = nengo.Ensemble(
      n_neurons=int(n_nrns*d/4),
      dimensions=d,
      max_rates=nengo.dists.Uniform(80, 120),
      intercepts=nengo.dists.Uniform(-1, 0.5),
      neuron_type=nengo.SpikingRectifiedLinear(),
      radius=0.5
    )
    otp = nengo.Node(size_in=2)

    nengo.Connection(inp, ldn.input, transform=B_p, synapse=0.1)
    nengo.Connection(ldn.output, ldn.input, transform=A_p, synapse=0.1)

    nengo.Connection(ldn.output, ens, synapse=0.010)
    np.random.seed(SEED)
    conn_mat = np.random.rand(2, d)
    nengo.Connection(ens, otp, transform=np.random.rand(2, d), synapse=0.010)

    net.p = nengo.Probe(otp)

  return net

if __name__ == "__main__":
  T = 140
  n_nrns = 100
  d = 6
  x = np.random.rand(T) - 1.0
  print(x, x.shape)

  A_p, B_p = get_A_p_and_B_p_matrices(d)

  net = get_nengo_slrc_model(x, A_p, B_p, d, n_nrns)

  with nengo_loihi.Simulator(net) as sim:
    sim.run(T/1000)

  print(sim.data[net.p], sim.data[net.p].shape)
