from SALib.analyze import sobol
import numpy as np

problem = {
  'num_vars': 7,
  'names': ['r','c','d','g','h','m','p'], #'k', 'a','b1', 'b2',
  'bounds': [[1.5, 10.5],[0, 0.05],[95, 225],[0, 0.02],[0, 120],[0.3, 12.3],[2.7,18]]
} # changed the bounds of the first two parameters to match the paper - is this right?

param_values = np.loadtxt('param_values.csv', delimiter=',')
Y = np.loadtxt('outputs.csv', delimiter=',')

# Perform analysis, calculate sensitivity indices
# The outputs are a problem - Y[:,0] is always zero, and Y[:,1] is ~10^-200
# this makes the sensitivity results not meaningful

Si = sobol.analyze(problem, Y[:,1], 
                  print_to_console=True, num_resamples = 1000, calc_second_order=False)

# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# (first and total-order indices with bootstrap confidence intervals)
