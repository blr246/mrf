import collections
import itertools as it
import mrf
import ve
from operator import mul
from operator import add
import numpy as np

def tree_sum_product(tree_network, root):
    '''
    Perform sum-product belief propagation for trees computing the marginal
    distribution of each variable using two passes.

    Parameters
    ----------
    tree_network : mrf.Network
        Network of factors over a tree graph structure.
    root : name
        Name of the variable to treat as the tree root. Can be of the network's
        variables.
    Returns
    -------
    marginals : dict of functions
        Map of functions keyed by variable name representing the marginal
        probability of each variable.
    '''

    assert not tree_network.is_energy_funcs

    def generate_message(cj, cij, incoming):

        # Create a summed out over the variable in cj.
        factors = reduce(add, incoming, []) + [cj, cij]
        psi = mrf.Network(factors)
        tau = ve.eliminate(psi, cj.names)

        # Make messages sum to one.
        xi_nstates = tau.nstates[0]
        alpha = np.sum(tau((xi,)) for xi in xrange(xi_nstates))
        tau = tau.partition(alpha)

        return tau.factors

    def generate_marginal(ci, messages):

        # Multiply out factors (all for the same variable).
        factors = reduce(add, messages, []) + [ci]
        marginal_table = reduce(mul, [fac.table for fac in factors])
        n = mrf.Network((mrf.Factor.fromTable(ci.names, marginal_table),))
        alpha = sum(n.query(s)
                for s in it.product(*[range(i) for i in n.nstates]))
        return mrf.Network(n.factors, alpha)

    return _tree_belief_propagation(
            tree_network, root, generate_message, generate_marginal)

def tree_max_product(tree_network, root):
    '''
    Perform max-product belief propagation for trees computing the MAP
    assignment to all of the tree variables.

    Parameters
    ----------
    tree_network : mrf.Network
        Network of factors over a tree graph structure.
    root : name
        Name of the variable to treat as the tree root. Can be of the network's
        variables.
    Returns
    -------
    max_marginal : dict of functions
        Map of functions keyed by variable name whose argmax represents the MAP
        assignment of the corresponding variable.
    '''

    assert not tree_network.is_energy_funcs

    def generate_message(cj, cij, incoming):
        ''' Generate a normalized message. '''

        # Compute max over xj for the message to xi.
        if cj.names[0] == cij.names[0]:
            edge = lambda xi, xj: (xj, xi)
            xi_nstates = cij.table.shape[1]
        else:
            edge = lambda xi, xj: (xi, xj)
            xi_nstates = cij.table.shape[0]

        # Compute the total response for the states of xi.

        incoming_tab = np.array([
            mul(cj((xj,)), reduce(mul, (m(xj) for m in incoming), 1.))
            for xj in range(cj.nstates[0])])

        message = lambda xi: max(mul(cij(edge(xi, xj)), incoming_tab[xj])
                for xj in range(cj.nstates[0]))

        # Combine into a single table over states of xi.

        table = np.fromiter((message((xi,))
                for xi in xrange(xi_nstates)), dtype=float)
        np.divide(table, np.sum(table), table)

        return lambda xi: table[xi]

    def generate_marginal(ci, messages):
        ''' Return normalized product over this variable and its messages. '''

        # Compute the total response for the states of xi.

        xi_nstates = ci.nstates[0]

        table = np.fromiter(
                (mul(ci((xi,)), reduce(mul, (m(xi) for m in messages), 1.))
                    for xi in xrange(xi_nstates)), dtype=float)
        np.divide(table, np.sum(table), table)

        return lambda xi: table[xi]


    return _tree_belief_propagation(
            tree_network, root, generate_message, generate_marginal)

def tree_max_sum(tree_network, root):
    '''
    Perform max-sum belief propagation for trees computing the MAP assignment
    to all of the tree variables.

    Parameters
    ----------
    tree_network : mrf.Network
        Network of factors over a tree graph structure.
    root : name
        Name of the variable to treat as the tree root. Can be of the network's
        variables.
    Returns
    -------
    max_marginal : dict of functions
        Map of functions keyed by variable name whose argmax represents the MAP
        assignment of the corresponding variable.
    '''

    assert tree_network.is_energy_funcs

    def generate_message(cj, cij, incoming):
        ''' Compute max-sum message from xi to xj. '''

        # Compute max over xj for the message to xi.

        if cj.names[0] == cij.names[0]:
            edge = lambda xi, xj: (xj, xi)
        else:
            edge = lambda xi, xj: (xi, xj)

        incoming_tab = np.array([
            add(cj((xj,)), reduce(add, (m(xj) for m in incoming), 0.))
            for xj in range(cj.nstates[0])])

        # Memoize message.
        cache = dict()

        def message(xi):
            if xi not in cache:
                cache[xi] = max(
                        add(cij(edge(xi, xj)), incoming_tab[xj])
                        for xj in range(cj.nstates[0]))
            return cache[xi]

        return message

    def generate_marginal(ci, messages):
        ''' Return the product over this variable and its messages. '''

        return lambda xi: \
            add(ci((xi,)), reduce(add, (m(xi) for m in messages), 0.))

    return _tree_belief_propagation(
            tree_network, root, generate_message, generate_marginal)

def tree_network_map_assignment(tree_network):
    '''
    Compute the MAP assignment for a tree mrf.

    Parameters
    ----------
    tree_network : mrf.Network
        Network of factors over a tree graph structure.
    Returns
    -------
    map_query : dict
        Map keyed by variable name giving a tuple of (potential, state) for
        each variable in the network.
    '''

    if tree_network.is_energy_funcs:
        max_marginals = tree_max_sum(tree_network, tree_network.names[0])
    else:
        max_marginals = tree_max_product(tree_network, tree_network.names[0])

    map_query = dict()
    # Iterate over single node potentials.
    for fac in tree_network.factors:
        if len(fac.names) > 1:
            continue

        name, nstates = fac.names[0], fac.nstates[0]
        max_marginal = max_marginals[name]
        p_map = [max_marginal(i) for i in range(nstates)]
        map_argmax = np.argmax(p_map)
        map_query[name] = (p_map[map_argmax], map_argmax)

    return map_query

def _tree_belief_propagation(
        tree_network, root, generate_message, generate_marginal):
    '''
    Internal method for message passing in two passes.
    '''

    # For a tree network, all cliques are over pairs (edges) or single nodes.
    t = mrf.network_to_mrf(tree_network)

    # Initialize cliques.
    cliques = dict(
            ((tuple(sorted(fac.names)), fac) for fac in tree_network.factors))

    # Find message passing order up to the root using DFS.
    pass_up = {}
    to_visit = [(root, t.edge[root])]
    while len(to_visit) > 0:

        # Get next set to visit
        xj, xis = to_visit.pop()

        # Visit over each edge.
        for xi in xis:

            if not xi in pass_up and xi != root:
                pass_up[xi] = xj
                to_visit.append((xi, t.edge[xi]))

    # Create dict of messages remaining and list of ready cliques.
    messages_remain = dict(((n, d-1) for n,d in t.degree().iteritems()))
    messages_remain[root] = t.degree()[root]

    # While there is a ready node.
    ready = [n for n,c in messages_remain.iteritems() if c is 0]
    messages = collections.defaultdict(lambda: {})
    while len(ready) > 0:

        # Pop the next clique to process; break on root.
        xi = ready.pop()

        # Get parent node (must be unique for a tree).
        xj = pass_up[xi]

        # Store the message xi -> xj.
        edge = tuple(sorted((xi,xj)))
        assert xj not in messages[xi]
        assert xi not in messages[xj]
        messages[xj][xi] = generate_message(
                cliques[(xi,)], cliques[edge], messages[xi].values())

        # Account for xj's new message.
        messages_remain[xj] -= 1
        assert messages_remain[xj] >= 0
        if 0 == messages_remain[xj] and xj != root:
            ready.append(xj)

    # No messages remain.
    assert sum(messages_remain.values()) == 0

    # Compute pass down adjacency list.
    pass_down = collections.defaultdict(lambda: [])
    for k,v in pass_up.iteritems():
        pass_down[v].append(k)

    # Pass down.
    marginals = {}
    ready = [root]
    while len(ready) > 0:

        xi = ready.pop()

        # Gather marginal.
        marginals[xi] = generate_marginal(
                cliques[(xi,)], messages[xi].values())

        for xj in pass_down[xi]:

            edge = tuple(sorted((xi,xj)))
                
            messages_sub_xj = [
                    messages[xi][xk] for xk in messages[xi].keys() if xk != xj]

            # Store the message xi -> xj.
            assert xi not in messages[xj]
            messages[xj][xi] = generate_message(
                    cliques[(xi,)], cliques[edge], messages_sub_xj)

            ready.append(xj)

    # Return marginal distributions for all variables.
    return marginals, messages

