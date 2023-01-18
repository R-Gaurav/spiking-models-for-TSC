import nengo
import nengo_loihi

class RTC(object):
  # LMU Constants at Run Time.
  ################################################################################
  CONFIG = 3 # Changet this config value everytime a change in parameters is done.
  SIMULATOR = "sim_nloihi" # One of "sim_nloihi" or "sim_nengo".
  BKND = "sim" # One of "sim" or "loihi" to be used with nengo_loihi.Simulator().
  ################################################################################

  # Network Parameters.
  ################################################################################
  N_SPK_NEURONS = 100 # Number of spiking neurons in each Ensemble.
  SEED = 9 # A random and fixed seed for reproducibility. #10 and #9.
  SYNAPSE = 0.010 # Synaptic filter time constant for all the connections.
  LOIHI_MIN_RATE = 80 # Minimum firing rate of the Ensemble neurons on Loihi.
  LOIHI_MAX_RATE = 120 # Maximum firing rate of the Ensemble neurons on Loihi.
  ################################################################################

  # RES Parameters
  ################################################################################
  #LDN_NEURON_TYPE = nengo_loihi.neurons.LoihiSpikingRectifiedLinear()
  LDN_NEURON_TYPE = nengo.SpikingRectifiedLinear()
  LDN_RADIUS = 1.5
  ORDER = 6 # Number of Legendre Polynomials to represent delay window. # 8
  THETA = 0.12 # Length of delay window LMU is supposed to represent, in seconds.
  TAU = 0.1 # The synaptic filter for LMU connections.
  # Default Low/High Intercepts -1, 0.5 in EnsembleArray
  ITRCPT_LOW = -1.0
  ITRCPT_HGH = 0.5
  ################################################################################

  # Signal ENS Parameters..
  ################################################################################
  #SIG_ENS_NEURON_TYPE = nengo_loihi.neurons.LoihiSpikingRectifiedLinear()
  SIG_ENS_NEURON_TYPE = nengo.SpikingRectifiedLinear()
  SIG_ENS_RADIUS = 0.5
  SIG_ENS_ITRCPT_LOW = -1.0
  SIG_ENS_ITRCPT_HGH = 0.5
  ################################################################################

  # RES Inhibition parameters.
  ################################################################################
  INHIBIT_CONST = 8.0 # At 0, no inhibition, minimum 2 requried, higher the better.
  INHIBIT_DURTN = 50 # Increasing this doesn't make the extracted Leg Sigs better.
  ################################################################################

  # Number of training, training evaluation, and test samples.
  NUM_TRAINS = 500
  NUM_TESTS = 4500
