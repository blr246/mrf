"""
Use the Max-Product Linear Programming technique to find the MAP assignment.

Data files taken from:
    http://www.hlt.utdallas.edu/~vgogate/uai14-competition/information.html
"""
import os

import mrf

__SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(__SCRIPT_DIR, 'data')

for path in (os.path.join(DATA_DIR, filename)
             for filename in os.listdir(DATA_DIR) if filename.endswith('.uai')):
    with open(path, 'r') as uaifile:
        network = mrf.uai.load(uaifile)
    with open('{}.evid'.format(path), 'r') as evidfile:
        network, evidence = mrf.uai.with_evidence(network, evidfile)
    network = network.to_energy_funcs()
    x_best, x_best_iter = mrf.mplp(network)
    print "Estimated MAP:", x_best
