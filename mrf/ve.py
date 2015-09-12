'''
Brandon L. Reiss

ve - A graphical models variable elimination library.
'''
import itertools as its
import networkx as nx

import numpy as np

import mrf


def _combine_factors(network):
    ''' Combine factors over identical variables. '''

    vars_to_fac_idx = {}
    combined_factors = []
    if network.is_energy_funcs:
        combine_op = np.add
    else:
        combine_op = np.multiply
    for f in network.factors:
        key = tuple(sorted(f.names))
        if key in vars_to_fac_idx:
            f_combined = combined_factors[vars_to_fac_idx[key]]
            f_table = f.table
            # Swap any axes that are not aligned to the combined view.
            for idx, (var_combined, var) in enumerate(
                    its.izip(f.names, f_combined.names)):
                if var_combined != var:
                    idx_swap = (i for i in xrange(idx, len(f.names))
                                if f.names[i] == var_combined).next()
                    f_table = np.swapaxes(f_table, idx, idx_swap)
            # Combine the tables now that vars are aligned.
            combine_op(f_combined.table, f_table, f_combined.table)
        else:
            idx = len(combined_factors)
            # Copy the table for the combined factor.
            combined_factors.append(mrf.Factor.fromTable(f.names,
                                                         np.copy(f.table)))
            vars_to_fac_idx[key] = idx

    return mrf.Network(combined_factors,
                       alpha=network.alpha,
                       names_order=network.names,
                       is_energy_funcs=network.is_energy_funcs)


def condition_eliminate(network, scope, evidence, order, normalize=False):
    '''
    Compute a network that answers conditional queries P(scope | evidence).

    Parameters
    ----------
    network : mrf.Network
        Network to condition and eliminate.
    scope : list of string
        List of variables to query after conditioning on evidence.
    evidence : dict or list
        Set of variables with fixed state used for conditioning.
    order : list of string
        List of variable names in elimination order.
    normalize : bool
        Flag indicating whether or not to re-partition the network.
    Returns
    -------
    network_cond_ve : mrf.Network
        Network resulting from conditioning and elimination. The network will
        not be a proper distrubution unless normalize=True.
    '''

    scope = set(scope)
    evidence_vars = set(evidence)

    # Check that evidence and scope are subsets of the network.
    if not scope.issubset(set(network.names)):
        raise ValueError("Observed is not a subset of the network")
    if not evidence_vars.issubset(set(network.names)):
        raise ValueError("Evidence is not a subset of the network")
    # Check that evidence and scope variables are disjoint.
    if not scope.isdisjoint(evidence_vars):
        raise ValueError(("Observed and evidence must be disjoint; found " +
                          "{} in intersection").format(
                              scope.intersection(evidence_vars)))

    elim = set(network.names) - scope - evidence_vars
    elim_ordered = [v for v in order if v in elim]
    if len(elim) != len(elim_ordered):
        raise ValueError(
            "Elimination order missing eliminated variables {}".format(
                list(elim - set(order))))

    # Condition all factors.
    n_given_e = [f.given_evidence(evidence)
                 for f in network.factors
                 if not set(f.names).issubset(evidence_vars)]
    names_order = [name for name in network.names if name not in evidence_vars]
    n_cond = mrf.Network(n_given_e,
                         alpha=network.alpha,
                         names_order=names_order,
                         is_energy_funcs=network.is_energy_funcs)

    # Perform variable elimination on the set of non-evidence and non-scope
    # variables.
    if len(elim_ordered) > 0:
        n_cond_elim = eliminate(n_cond, elim_ordered)
    else:
        n_cond_elim = _combine_factors(n_cond)

    if normalize:
        return n_cond_elim.partition()
    else:
        return n_cond_elim


def eliminate(network, elim):
    '''
    Perform variable elimination on network using the given order.

    Parameters
    ----------
    network : mrf.Network
        Network to eliminate.
    elim : list of string
        List of variable names to eliminate in elimination order.
    Returns
    -------
    network_ve : mrf.Network
        Network resulting from elimination.
    '''

    if len(elim) == 0:
        return network

    def elim_slice_generator(factor, psi_vars, elim_var):
        '''
        Generate a function : permutation -> slice that extracts the slice of
        the factor table relevant to the states of the eliminated variable with
        the rest of the variables fixed in the scope of the factor.
        Parameters
        ----------
        factor : mrf.Factor
            The mrf.Factor from which to eliminate a variable.
        psi_vars : list of string
            The variables in the full factor.
        elim_var : string
            The Variable to eliminate.
        Returns
        -------
        Function from an assignment to psi_vars to the relevant slice of the
        factor over the states of elim_var.
        '''

        psi_var_indices = dict((var, idx)
                               for var, idx in its.izip(psi_vars, its.count()))

        def query_factor(assignment):
            # Reorder assignment in factor scope replacing elim_var's state with
            # slice(None) to capture the full dimension.
            to_factor_scope = [assignment[psi_var_indices[var]]
                               if var != elim_var else slice(None)
                               for var in factor.names]
            return factor.table[to_factor_scope]
        return query_factor

    is_energy_funcs = network.is_energy_funcs
    network = network.to_linear_funcs()

    elim = set(elim)
    nstates_per_var = dict(its.izip(network.names, network.nstates))
    network_ve = list(network.factors)

    # For each elimination variable, perform sum-product elimination.
    alpha = 1.
    for v in elim:

        # Locate all factors f containing v in Scope(f).
        f_contains_v = [f for f in network_ve if v in f]
        # Remove factors with eliminated variable in scope.
        network_ve = [f for f in network_ve if v not in f]

        # Get full set of variables to accumulate new table into.
        psi_vars_set = set(
            its.chain.from_iterable((f.names for f in f_contains_v))) - {v}
        psi_vars = list(var for var in network.names if var in psi_vars_set)

        # When the factor is empty, use it as a constant.
        if len(psi_vars) == 0:
            alpha *= np.sum(f.table)
            continue

        # Gather slice generators.
        factor_slice_generators = [elim_slice_generator(factor, psi_vars, v)
                                   for factor in f_contains_v]

        # Create a factor table to sum into.
        psi_nstates = tuple([nstates_per_var[vname] for vname in psi_vars])
        table = np.ones(psi_nstates)

        # For each permutation of the variables in the state, sum the factors
        # sliced to the scope of the eliminated variable.
        def reduce_states(accum, table):
            return np.multiply(table, accum, accum)

        v_accum = np.empty(nstates_per_var[v])
        for perm in its.product(*[range(n) for n in psi_nstates]):
            v_accum[:] = 1.
            table[perm] = np.sum(
                reduce(reduce_states,
                       (v_slice(perm) for v_slice in factor_slice_generators),
                       v_accum))

        # Add new factor.
        network_ve.append(mrf.Factor.fromTable(psi_vars, table))

    # Finally, return a Network of the remaining factors.
    names_order = [name for name in network.names if name not in elim]
    network_ve = mrf.Network(network_ve,
                             alpha=network.alpha * alpha,
                             names_order=names_order,
                             is_energy_funcs=False)
    network_ve = _combine_factors(network_ve)
    if is_energy_funcs:
        return network_ve.to_energy_funcs()
    else:
        return network_ve


def greedy_ordering(mrf, score_func, with_rand=True):
    '''
    Remove node with the lowest score adding fill edges.

    Parameters
    ----------
    mrf : graph
        Undirected graph representing network to order for variable
        elimination.
    score_func : function
        A function (network.Graph, node) -> int that ranks nodes for variable
        elimination ordering.
    with_rand : bool
        Flag to enable or disable random perturbation for nodes taking equal
        scores.
    Returns
    -------
    order : list
        Removal order of nodes by name using score function heuristic.
    g : network.Graph
        Induced graph resulting from removal order.
    '''

    # Initialize list to remove and sort repeatedly to find min score.
    to_remove = mrf.nodes()
    to_remove_set = set(mrf.nodes())
    # Copy graph for adding edges during ordering.
    g = nx.Graph(mrf)
    induced = nx.Graph(mrf)

    # Create score function closed over the graph.
    # Add within interval [0,1) to randomize equal scores.
    def score(n):
        return score_func(g, n)

    if with_rand:
        def score_tie(n):
            return score_func(g, n) + np.random.uniform()
    else:
        score_tie = score

    order = []
    for i in range(mrf.number_of_nodes()):
        # Sort the removal list based on score function.
        to_remove.sort(key=score_tie, reverse=True)
        # Remove the element with the minimum score.
        removed = to_remove.pop()
        min_score = score(removed)
        to_remove_set.remove(removed)
        order.append((removed, min_score))
        # Add fill edges for the removed element.
        clique_nodes = [removed] \
            + [n for n in g.neighbors(removed) if n in to_remove_set]
        g.add_edges_from((e for e in its.combinations(clique_nodes, 2)))
        induced.add_edges_from((e for e in its.combinations(clique_nodes, 2)))
        g.remove_node(removed)

    return order, induced


def min_fill(g, n):
    ''' Compute fill edges for a node belonging to the given graph. '''

    # Get a subgraph induced on [n] + [Nh(n)] and count edges.
    clique_nodes = [n] + g.neighbors(n)
    current_edges = g.subgraph(clique_nodes).number_of_edges()
    max_edges = (len(clique_nodes) * (len(clique_nodes) - 1)) / 2
    return max_edges - current_edges
