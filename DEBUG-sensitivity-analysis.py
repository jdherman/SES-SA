from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from simulate import simulate_SES
import pyDOE as doe


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
    print('Run %d: Eq: %d, Pop: %0.2f' % (n, eq_condition[n], U_array[n]))

  resilience = np.sum(eq_condition) / num_points # proportion of states that lead to non-collapse equilibrium
  equity = np.mean(U_array) # total well being  

  print('Res: %0.2f, Eq: %0.2f' % (resilience, equity))
  return resilience, equity

# Paper values
x = [10.0, 0.05, 100, 0.01, 0.06, 0.8, 3]
# Y = SES_model(x)
# [5.72753906e+00 4.23339844e-02 1.29912109e+02 8.53515625e-03
#  5.85937500e-01 8.18671875e+00 1.48772461e+01]
b = [(0.50*i, 1.50*i) for i in x]
problem = {
  'num_vars': 7,
  'names': ['r','c','d','g','h','m','p'], #'k', 'a','b1', 'b2',
  'bounds': [[5,10.5],[0, 0.05],[95, 175],[0, 0.015],[0, 120],[0.3, 1.0],[2.7,3.3]]
} # changed the bounds of the first two parameters to match the paper - is this right?

# Generate samples
param_values = saltelli.sample(problem, 1000, calc_second_order=False)
i = np.random.randint(len(param_values))
# i = np.argmin(np.mean((param_values - x)**2)**2)
print(i)
print(param_values[i])
Y = SES_model(param_values[i])

# Run model
# for i in range(N):
  # Y[i,:] = SES_model(param_values[i])
  # print(Y[i,:])
