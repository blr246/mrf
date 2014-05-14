'''
Max-product Linear Programming (MPLP) algorithm for finding the MAP assignment
to a Markov Random Field.

(c) 2013 Brandon L. Reiss

Written for
  Probabilistic Graphical Models
  taught by Prof. David Sontag at NYU
  CSCI-GA.3033 Spring 2013
'''
import itertools as it
import mrf
import numpy as np

def mplp(n, eps=2e-4, max_iter=1e3, momentum=0.):
    '''
    Perform Max-Product Linear Programming on a network of energy functions to
    find the MAP assignment.

    Parameters
    ----------
    n : mrf.Network
        A network of energy functions.
    eps : number
        Termination threshold for sequential difference in the loss function.
    max_iter : integer
        Max number of iterations.
    momentum : number
        Momentum applied to message updates as in
        msg_t = (momentum * mst_(t-1)) + ((1 - momentum) * msg_new).
    Returns
    -------
    x_best : list
        List of assignments to network variables in the order of n.names. To
        compute the value of the MAP assignment, one would execute n(x_best).
    x_best_iter : integer
        Index of the iteration when x_best was found.
    '''

    if momentum < 0 or momentum >= 1:
        raise ValueError("Must have 0 < momentum <= 1")

    assert(n.is_energy_funcs)
    g = mrf.network_to_mrf(n)

    name_to_idx_map = dict((p for p in zip(n.names, it.count())))
    name_to_idx = lambda v: name_to_idx_map[v]

    single_node_factors = [fac for fac in n.factors if len(fac.names) == 1]
    single_node_factors = sorted(
            single_node_factors, key=lambda f: name_to_idx(f.names[0]))

    edge_factors = [fac for fac in n.factors if len(fac.names) == 2]
    
    edges = dict(((name_to_idx(i), [name_to_idx(j) for j in js])
        for i,js in g.edge.iteritems()))

    # Initialize delta variables and edge factor map.
    delta = {}
    edge_factor_map = {}
    edges_order = []
    for fac in edge_factors:
        ij = tuple(map(name_to_idx, fac.names))
        edges_order.append(ij)
        ji = (ij[1], ij[0])
        delta[ij] = np.zeros(n.nstates[ij[1]])
        delta[ji] = np.zeros(n.nstates[ji[1]])

        edge_factor_map[ij] = fac

    # Initialize last loss and best MAP values.
    ell_last, p_xk_best = float('inf'), float('-inf')

    # Pass messages until "small enough change".
    iter_count = it.count()
    m, minv = momentum, (1. - momentum)
    while True:

        iter_curr = iter_count.next()

        # Compute new loss.
        ell_delta, x_new = _ell(
                delta, single_node_factors, edges, edge_factor_map)
        p_xk = n(x_new)

        if p_xk > p_xk_best:
            x_best, x_best_iter = x_new, iter_curr
            p_xk_best = p_xk

        ell_change = ell_delta - ell_last
        integrality_gap = ell_delta - p_xk

        print ('Iter: {0:2d}, L(d): {1:1.4e}, ' +
                'change_L(d): {2:1.4e}, P(x): {3:1.4e}, ' +
                'integrality_gap: {4:1.4e}').format(
                iter_curr, ell_delta, ell_change,
                p_xk, integrality_gap)

        ell_last = ell_delta

        if iter_curr >= max_iter or ell_change > -eps:
            break

        # Helper function for computing sum over messages that are fixed for
        # the current update.
        di_j = lambda i, j: \
                single_node_factors[i].table + \
                np.sum((delta[(k, i)] for k in edges[i] if k != j), axis=0)

        # Perform message updates.
        for ij in edges_order:

            edge = edge_factor_map[ij]
            ji = (ij[1], ij[0])

            dji_new = -0.5 * di_j(*ij) + 0.5 * np.max(
                    edge.table   + di_j(*ji), axis=1)
            assert(len(dji_new) == edge.nstates[0])

            dij_new = -0.5 * di_j(*ji) + 0.5 * np.max(
                    edge.table.T + di_j(*ij), axis=1)
            assert(len(dij_new) == edge.nstates[1])
            
            delta[ji] = (minv * dji_new) + (m * delta[ji])
            delta[ij] = (minv * dij_new) + (m * delta[ij])

    return x_best, x_best_iter

def _ell(delta, single_node_factors, edges, edge_factor_map):
    '''
    MPLP loss function. Performs local decoding of the MAP assignment.

    Parameters
    ----------
    delta : dict
        Map of (i,j) pairs to their messages delta_(i->j)(xj).
    single_node_factors : list
        List of mrf.Factor for all single-node factors.
    edges : dict
        Edge adjacency lists.
    edge_factor_map : dict
        Map of (i,j) pairs to their pairwise edge factors.
    Returns
    -------
    loss : number
        Value of the loss function for the current value of delta.
    x : list
        Joint assignment estimating the MAP assignment to the variables in the
        oder of the single node factors.
    '''

    # Local decoding.
    x = [None] * len(single_node_factors)

    # Single node potentials.
    sum_loss_node_factors = 0.
    for i,fac in enumerate(single_node_factors):

        sum_delta_ji = np.sum(
                (delta[(j, i)] for j in edges[i]), axis=0)
        loss_per_xi = fac.table + sum_delta_ji
        x[i] = np.argmax(loss_per_xi)

        sum_loss_node_factors += np.max(loss_per_xi)

    # Edge potentials.
    sum_loss_edge_factors = 0.
    for ij,edge in edge_factor_map.iteritems():

        ji = (ij[1], ij[0])

        # Compute max loss over all pairs (xi, xj).
        max_xixj = float('-inf')
        nstates_xi = len(delta[ji])
        for xi in range(nstates_xi):
            max_xixj = max(max_xixj,
                    np.max(edge.table[xi] - delta[ji][xi] - delta[ij]))

        sum_loss_edge_factors += max_xixj

    return sum_loss_node_factors + sum_loss_edge_factors, x

