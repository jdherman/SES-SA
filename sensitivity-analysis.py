from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from simulate import simulate_SES
import pyDOE as doe
from mpi4py import MPI
np.seterr(divide='ignore', invalid='ignore') # avoid warnings about div by zero in optimization

comm = MPI.COMM_WORLD

np.random.seed(1)

def SES_model(x):

  r,c,d,g,h,m,p = list(x)

  # other constants not being sampled
  W_min = 0
  R_max = 100  # aquifer capacity
  q = -0.5 #substituttion parameter to reflect limited substitutability between resource and labor
  dt = 0.08
  k = 2 # water access parameter (how steeply benefit of water rises)
  a = 1000  # profit cap
  b1 = 1 # profit vs. water parameter, or = 0.05
  b2 = 1 # profit vs. labor parameter, or = 0.05

  # set policy
  # fine = 130
  # fine_cap = 5

  num_points = 100 #Was 80
  initial_points = doe.lhs(3, samples = num_points)

  # Scale points ([R, U, W])
  initial_points[:,0] = initial_points[:,0] * 100
  initial_points[:,1] = initial_points[:,1] * 45
  initial_points[:,2] = initial_points[:,2] * 20

  # initialize matrix recording whether initial point leads to good or bad eq
  eq_condition = np.zeros(num_points)
  U_array = np.zeros(num_points)

  for n, point in enumerate(initial_points): 
    R_0 = point[0]
    U_0 = point[1]
    W_0 = point[2]

    # pp = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m, # With policy
    #                 W_min, dt, R_0, W_0, U_0, fine_cap, fine)
    # R, E, U, S, W, P, L, converged = pp.run()

    ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m, # No policy
                        W_min, dt, R_0, W_0, U_0)
    R, E, U, S, W, P, L, convergence = ts.run() # this is an array

    if R[-1] > 90 and U[-1] < 1:
      eq_condition[n] = 0
    else:
      eq_condition[n] = 1

    U_array[n] = U[-1] #np.mean(U) # population (pseudo for resilience)

  resilience = np.sum(eq_condition) / num_points # proportion of states that lead to non-collapse equilibrium
  equity = np.mean(U_array) # total well being  

  return resilience, equity


problem = {
  'num_vars': 7,
  'names': ['r','c','d','g','h','m','p'], #'k', 'a','b1', 'b2',
  'bounds': [[1.5, 10.5],[0, 0.05],[95, 225],[0, 0.02],[0, 120],[0.3, 12.3],[2.7,18]]
} # changed the bounds of the first two parameters to match the paper - is this right?

# Generate samples
param_values = saltelli.sample(problem, 10000, calc_second_order=False)
N = len(param_values) # 10000 * (k+2) = 90000 total model runs
model_runs_per_proc = int(N / comm.size)
start_index = model_runs_per_proc * comm.rank
end_index = model_runs_per_proc * (comm.rank + 1)

# Run SES Model for each parameter set, save output
Y = np.zeros((model_runs_per_proc,2))

# Run model
for i in range(start_index, end_index):
  print('Processor %d: Running index %d' % (comm.rank, i), flush=True)
  Y[i-start_index,:] = SES_model(param_values[i])

comm.Barrier() # wait for all of them to finish
Y_all = comm.gather(Y, root=0) # gather back to master node (0)

# the combined results only exist on node 0, save from there
if comm.rank==0:
  # they gather into lists of arrays, unfortunately. fix that:
  Y_all = np.array(Y_all).reshape((N, 2))
  np.savetxt('param_values.csv', param_values, delimiter=',')
  np.savetxt('outputs.csv', Y, delimiter=',')


# # Perform analysis, calculate sensitivity indices
# Si = sobol.analyze(problem, Y[:,0], 
#                   print_to_console=True, num_resamples = 100) #1000
# # Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# # (first and total-order indices with bootstrap confidence intervals)

# Sii = sobol.analyze(problem, Y[:,1], 
#                   print_to_console=True, num_resamples = 100) #1000