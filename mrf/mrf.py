'''
Brandon L. Reiss

mrf - A general Markov Random Field library.
'''
import numpy as np
import itertools as it
import string
import operator
import networkx as nx

def query_table(names, nstates, q, table):
    '''
    Query the potential function defined by the N-dimensional matrix
    table over a full query assignment.

    Parameters
    ----------
    names : list of string
        List of variable names.
    nstates : list of int
        List of state counts for each variable.
    q : query
        Variable states for all variables in the joint distribution. If the
        query is not a dict, then its iteration order is assumed to match
        the canonical ordering of Factor.names.
    table : array
        Array of potential responses for joint assignments of the variables.
    Returns
    -------
    p : number
        Probability P(X1=x1,...,XN=xn).
    '''

    # Execute query based on type.
    if isinstance(q, dict):
        # Query map contains all variables in range.
        for v,s in it.izip(names, nstates):
            if not v in q:
                raise ValueError(("Missing variable {}; the query map " +
                        "must contain all names").format(v))
            if not q[v] >= 0 and q[v] < s:
                raise ValueError(("Assignment {}={} is out of range for " +
                    "ntates={}; variables must be in range " +
                    "of nstates").format(v, q[v], s))

        # Query is valid.
        return table[tuple([q[v] for v in names])]
    else:
        # Query list contains all variables in range.
        if len(q) != table.ndim:
            raise ValueError(("len(q)={}, N={}; query " +
                "must have len(q)=N)").format(len(q), table.ndim))

        for qs,v,s in it.izip(q, names, nstates):
            if not qs >= 0 and qs < s:
                raise ValueError(("Assignment {}={} is out of range for " +
                    "ntates={}; variables must be in range " +
                    "of nstates").format(v, qs, s))

        # Query is valid.
        return table[tuple(q)]

class Factor(object):
    '''
    Potential over a clique of nodes in a PGM represented as a full
    N-dimensional table.
    '''

    @staticmethod
    def fromTable(names, table):
        return Factor(names, table=table)

    @staticmethod
    def fromVariable(var):
        return Factor(([cvar.name for cvar in var.cond_vars] + [var.name] if
                var.cond_vars else [var.name]), var=var)

    def __init__(self, names, table=None, var=None):
        '''
        Initialize the factor from different input data. The data are not
        copied. See fromXxx() static functions.
        '''

        self._names = list(names)
        self._names_set = set(names)

        # Initialize from full table of potentials.
        if table is not None:
            assert var is None
            self._table = table

        # Initialize from a Variable.
        if var is not None:
            assert table is None
            self._table = var.probs

        if len(names) != self.table.ndim:
            raise ValueError("len(names)={} must match ndim={}".format(
                len(names), self.table.ndim))

    def __call__(self, *args):
        return self.query(*args)

    def __repr__(self):
        return 'fac(' + string.join(self.names, ', ') + ')'

    def contains_var(self, var):
        return var.name in self._names_set

    def contains_var_by_name(self, name):
        return name in self._names_set

    @property
    def names(self):
        return self._names

    @property
    def nstates(self):
        return self._table.shape

    @property
    def table(self):
        return self._table

    def given_evidence(self, evidence):
        '''
        Assign variables in evidence to fixed values and return a new factor
        that is the current factor given the evidence.

        Parameters
        ----------
        evidence : dict
            Map of assigned variables. The evidence does not have to be
            comprehensive in the scope of this factor.
        Returns
        -------
        factor : Factor
            New factor with variables in this scope fixed as per evidence.
        '''

        slicer = map(
                lambda v: evidence[v] if v in evidence else slice(None),
                self.names)
        names = [v for v in self.names if not v in evidence]
        if len(names) > 0:
            return Factor.fromTable(names, self.table[slicer])
        else:
            return None

    def query(self, q):
        return query_table(self.names, self.nstates, q, self.table)


class Network(object):
    '''
    A group of factors and a valid partitioning constant defines a network.
    Network queries are over the set of free variables for the network. Note
    that if the factors encode conditioning, then the free variables are
    implicitly conditioned upon the state of the factors. Such annotations are
    not exposed by this structure.
    '''

    def __init__(self, factors, alpha=1.0,
            names_order=None, is_energy_funcs=False):

        # Initialize name order.
        all_names = set(it.chain(*[f.names for f in factors]))
        if names_order is None:
            names_order = list(all_names)
        assert set(names_order) == all_names

        # Get variable states per name.
        name_to_idx = dict(
                ((name,idx) for name,idx in it.izip(names_order, it.count())))
        nstates_ordered = [None] * len(names_order)
        for f in factors:
            for name,nstates in it.izip(f.names, f.nstates):
                idx = name_to_idx[name]
                if nstates_ordered[idx] is not None:
                    assert nstates_ordered[idx] == nstates
                else:
                    nstates_ordered[idx] = nstates

        # Setup query.
        if not is_energy_funcs:
            self.query = self._query_prod
        else:
            self.query = self._query_sum

        # Gather factor variables.
        self._names = names_order
        self._nstates = nstates_ordered
        self._alpha = alpha
        self._partition = 1.0 / alpha
        self._factors = factors

    def __repr__(self):
        return "net: F(" + string.join(self.names, ', ') + ')'

    def __call__(self, *args):
        return self.query(*args)

    @property
    def names(self):
        return self._names

    @property
    def nstates(self):
        return self._nstates

    @property
    def factors(self):
        return self._factors

    @property
    def alpha(self):
        return self._alpha

    @property
    def is_energy_funcs(self):
        return self.query == self._query_sum

    def partition(self, alpha):
        '''
        Apply an inverse partitioning constant alpha to the network. The
        current inverse partitioning constant remains intact so that the
        network returned has a sum over its joint distribution scaled by
        (1. / alpha).
        '''

        return Network(self.factors, self.alpha * alpha, self.names)

    def to_energy_funcs(self):
        '''
        Convert potentials to energy functions -ln(phi).
        '''

        if self.is_energy_funcs:
            return Network(self.factors, names_order=self.names)
        else:
            energy_funcs = [Factor.fromTable(fac.names, np.log(fac.table))
                    for fac in self.factors]
            return Network(energy_funcs,
                    names_order=self.names, is_energy_funcs=True)

    def _query_prod(self, q):
        '''
        Compute a product of factors given the query assignments. See
        mrf.query_table() for more information.
        '''

        # Convert query to a map if it is not already. The map form of query()
        # simplifies extracting assignments for each factor. 
        if not isinstance(q, dict):
            q = dict((p for p in it.izip(self.names, q)))

        # Return the product of the factors normalized by alpha.
        return self._partition * reduce(operator.mul,
                map(lambda fac: fac(q), self.factors))

    def _query_sum(self, q):
        '''
        Compute a sum of factors given the query assignments. See
        mrf.query_table() for more information.
        '''

        # Convert query to a map if it is not already. The map form of query()
        # simplifies extracting assignments for each factor. 
        if not isinstance(q, dict):
            q = dict((p for p in it.izip(self.names, q)))

        # Return the product of the factors normalized by alpha.
        return self._partition * reduce(operator.add,
                map(lambda fac: fac(q), self.factors))


def network_to_dag(network):
    ''' Convert a ve Network to a networkx DiGraph. '''

    edges = [(c,f.names[-1]) for f in network.factors for c in f.names[:-1]]
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g

def network_to_mrf(network):
    ''' Convert a ve Network to a networkx Graph. '''

    edges = [(c,f.names[-1]) for f in network.factors for c in f.names[:-1]]
    g = nx.Graph()
    g.add_edges_from(edges)
    return g

def moralize_bayesian_network(bn):
    ''' Moralize a networkx DiGraph representing a Bayesian network. '''

    # Create undirected graph.
    mrf = bn.to_undirected()

    # Find v-structures (occur on nodes s.t. indegree > 2) to moralize.
    for n,d in bn.in_degree_iter():
        if d > 1:
            # Moralize unmarried parents (same sex couples welcome).
            for mom,dad in it.combinations(bn.predecessors(n), 2):
                mrf.add_edge(mom, dad)

    return mrf

def network_to_rooted_tree(network, root):
    '''
    Convert an mrf.Network to a rooted tree.

    Parameters
    ----------
    network : mrf.Network
        The network to convert to a rooted tree.
    root : name
        Name of the variable to treat as the root.
    Returns
    -------
    tree : networkx.DiGraph
        Tree for the network rooted at root.
    '''

    g = nx.Graph((tuple(f.names) for f in network.factors if len(f.names) > 1))
    return graph_to_rooted_tree(g, root)

def graph_to_rooted_tree(g, root):
    '''
    Convert a networkx Graph to a rooted tree.

    Parameters
    ----------
    g : networkx.Graph
        The graph to convert to a rooted tree.
    root : name
        Name of the variable to treat as the root.
    Returns
    -------
    tree : networkx.DiGraph
        Tree for the network rooted at root.
    '''

    t = nx.DiGraph()

    def graph_to_rooted_tree_helper(r):

        # Get edges from r.
        e = g.edge[r]
        for dst in e:
            if not dst in t.node:
                t.add_edge(r,dst)
                graph_to_rooted_tree_helper(dst)

    graph_to_rooted_tree_helper(root)

    return t

def visualize_rooted_tree(t, prog='dot'):
    '''
    Visualize tree graph using graphviz dot.

    Parameters
    ----------
    t : networkx.DiGraph
        The tree to visualize.
    Returns
    -------
    None
    '''

    nx.draw_graphviz(t, prog=prog, font_size=6, node_shape='s',
            edge_color='gray', node_size=500, node_color='c')

