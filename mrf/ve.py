'''
Brandon L. Reiss

ve - A graphical models variable elimination library.
'''
import mrf
import numpy as np
import itertools as it
import operator
import networkx as nx

def condition_eliminate(network, observed, evidence, order):
    '''
    Compute a network that answers conditional queries P(observed | evidence).

    Parameters
    ----------
    network : mrf.Network
        Network to condition and eliminate.
    observed : list of string
        List of variables to query after conditioning on evidence.
    evidence : dict or list
        Set of variables with fixed state used for conditioning.
    order : list of string
        List of variable names in elimination order.
    Returns
    -------
    network_cond_ve : mrf.Network
        Network resulting from conditioning and elimination.
    '''

    # Check for at least one observed, evidence variables.
    if len(observed) == 0:
        raise ValueError("Observed set is empty")
    if len(evidence) == 0:
        raise ValueError("Evidence set is empty")
    # Check that evidence and observed are subsets of the network.
    if not set(observed).issubset(set(network.names)):
        raise ValueError("Observed is not a subset of the network")
    if not set(evidence).issubset(set(network.names)):
        raise ValueError("Evidence is not a subset of the network")
    # Check that evidence and observed variables are disjoint.
    if not set(observed).isdisjoint(set(evidence)):
        raise ValueError(("Observed and evidence must be disjoint; found " +
            "{} in intersection").format(
                set(observed).intersection(set(evidence))))

    # Condition all factors.
    n_given_e = set([f.given_evidence(evidence) for f in network.factors])
    n_given_e.remove(None)
    names_order = [name for name in network.names
            if name in (set(network.names) - set(evidence))]
    n_pre_cond = mrf.Network(n_given_e, names_order=names_order)

    # Perform variable elimination on the set of non-evidence and non-observed
    # variables.
    elim = set(n_pre_cond.names).difference(set(observed))
    # Apply order to elim set.
    elim_ordered = [v for v in order if v in elim]
    n_cond = eliminate(n_pre_cond, elim_ordered)

    # Compute partition over joint states of observed variables.
    alpha = 0.
    for perm in it.product(*[range(n) for n in n_cond.nstates]):
        alpha += n_cond.query(dict(
            ((n,s) for n,s in it.izip(n_cond.names, perm))))

    return n_cond.partition(alpha)

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

    # Get states per variable for permuting over joint states of factors.
    nstates_per_var = dict(it.izip(network.names, network.nstates))

    # Make a copy of the input network for elimination.
    network_ve = network.factors

    def slice_lambda_generator(factor, psi_vars, v):
        '''
        Generate a function : permutation -> slice that extracts the slice of
        the factor table relevant to the states of the eliminated variable with
        the rest of the variables fixed in the scope of the factor.
        Parameters
        ----------
        factor : mrf.Factor
            The mrf.Factor from which to eliminate a variable.
        psi_vars : list of Variable
            The variables in the full factor.
        v : Variable
            The Variable to eliminate.
        Returns
        -------
        '''

        # Closure for psi_vars index.
        def create_get_idx(i): return lambda p: p[i]

        # Iterate over variables of the factor and generate a tuple of lambdas
        # that take the mapped index in a permutation for the full factor psi.
        psi_vars_func_map = dict((k, create_get_idx(v))
                    for k,v in it.izip(psi_vars, it.count()))
        psi_vars_func_map[v] = lambda p: slice(None)
        psi_vars_funcs = [psi_vars_func_map[vname] for vname in factor.names]
        return lambda p: factor.table[[f(p) for f in psi_vars_funcs]]

    # For each elimination variable, perform sum-product elimination.
    alpha = 1.
    for v in elim:

        # Locate all factors f containing v in Scope(f).
        f_contains_v = [f for f in network_ve if f.contains_var_by_name(v)]
        # Remove factors with eliminated variable in scope.
        network_ve = [f for f in network_ve if not f.contains_var_by_name(v)]

        # Get full set of variables to accumulate new table into.
        psi_vars = list(set(it.chain.from_iterable(
            (f.names for f in f_contains_v))))
        psi_vars.remove(v)

        # When the factor is empty, use it as a constant.
        if len(psi_vars) == 0:
            alpha *= np.sum(f.table)
            continue

        # Gather slice generators.
        factor_slice_generators = [slice_lambda_generator(factor, psi_vars, v)
                for factor in f_contains_v]

        # Create a factor table to sum into.
        psi_vars_nstates = tuple(
                [nstates_per_var[vname] for vname in psi_vars])
        table = np.empty(psi_vars_nstates)

        # For each permutation of the variables in the state, sum the factors
        # sliced to the scope of the eliminated variable.
        for perm in it.product(*[range(n) for n in psi_vars_nstates]):
            p = np.sum(reduce(
                operator.mul, (g(perm) for g in factor_slice_generators)))
            table[perm] = p

        # Add new factor.
        network_ve.append(mrf.Factor.fromTable(psi_vars, table))

    # Finally, return a Network of the remaining factors.
    names_order = [name for name in network.names
            if name in (set(network.names) - set(elim))]
    return mrf.Network(network_ve, alpha, names_order)

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
    f = lambda n: score_func(g, n)
    # Add within interval [0,1) to randomize equal scores.
    if with_rand:
        fp = lambda n: f(n) + np.random.uniform()
    else:
        fp = f

    order = []
    for i in range(mrf.number_of_nodes()):
        # Sort the removal list based on score function.
        to_remove.sort(key=fp, reverse=True)
        # Remove the element with the minimum score.
        removed = to_remove.pop()
        score = f(removed)
        to_remove_set.remove(removed)
        order.append((removed, score))
        # Add fill edges for the removed element.
        clique_nodes = [removed] + [n for
                n in g.neighbors(removed) if n in to_remove_set]
        g.add_edges_from((e for e in it.combinations(clique_nodes, 2)))
        induced.add_edges_from((e for e in it.combinations(clique_nodes, 2)))
        g.remove_node(removed)

    return order,induced

def min_fill(g, n):
    ''' Compute fill edges for a node belonging to the given graph. '''

    # Get a subgraph induced on [n] + [Nh(n)] and count edges.
    clique_nodes = [n] + g.neighbors(n)
    current_edges = g.subgraph(clique_nodes).number_of_edges()
    max_edges = (len(clique_nodes) * (len(clique_nodes) - 1)) / 2
    return max_edges - current_edges
