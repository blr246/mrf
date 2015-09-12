'''
Brandon L. Reiss

mrf - A general Markov Random Field library.
'''
import itertools as it
import operator

import networkx as nx
import numpy as np


def _query_table(names, nstates, q, table):
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
        Initialize the factor from different input data.
        See also fromXxx() static functions.

        N.B. The table data are not copied defensively.
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

    def __contains__(self, name):
        return name in self._names_set

    def __repr__(self):
        prefix = 'Factor('
        base_indent = '\n' + ' ' * len(prefix)
        table_prop_offset = ' ' * len('table=')
        table_repr = (base_indent + table_prop_offset).join(
            repr(self.table).split('\n'))
        return '{}{})'.format(
            prefix,
            (',' + base_indent).join(
                [repr(self.names),
                 "table={}".format(table_repr)]))

    def contains_var(self, name):
        return name in self

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

        slicer = [evidence[v] if v in evidence else slice(None)
                  for v in self.names]
        names = [v for v in self.names if v not in evidence]
        if len(names) > 0:
            return Factor.fromTable(names, self.table[slicer])
        else:
            raise ValueError("Cannot condition on every variable; use query().")

    def query(self, *args):
        '''
        Query the factor with the given variable states.

        Note that the query can be individual variable states as in
            factor(0, 1, ...)
        or a single tuple of states as in
            factor((0, 1, ...))
        of a dictionary of states as in
            factor({'var1': 0, ...}).
        '''
        if len(args) == 0:
            raise ValueError("Factor query cannot be empty.")
        elif len(args) == 1:
            q = args[0]
            # When the query is a single value, wrap it in an iterable type.
            try:
                iter(q)
            except TypeError:
                q = (q,)
        else:
            q = args
        return _query_table(self.names, self.nstates, q, self.table)


class Network(object):
    '''
    A group of factors and a valid partitioning constant defines a network.
    Network queries are over the set of free variables for the network. Note
    that if the factors encode conditioning, then the free variables are
    implicitly conditioned upon the state of the factors. Such annotations are
    not exposed by this structure.
    '''

    def __init__(self, factors, alpha=None,
                 names_order=None, is_energy_funcs=False):

        # Initialize name order.
        all_names = set(it.chain(*[f.names for f in factors]))
        if names_order is None:
            names_order = list(all_names)
        assert set(names_order) == all_names

        # Get variable states per name.
        name_to_idx = dict((name, idx)
                           for name, idx in it.izip(names_order, it.count()))
        nstates_ordered = [None] * len(names_order)
        for f in factors:
            for name, nstates in it.izip(f.names, f.nstates):
                idx = name_to_idx[name]
                if nstates_ordered[idx] is not None:
                    assert nstates_ordered[idx] == nstates
                else:
                    nstates_ordered[idx] = nstates

        # Gather factor variables.
        self._names = names_order
        self._nstates = nstates_ordered
        if is_energy_funcs:
            # When log-linear, partition is a subtraction.
            if not alpha:
                alpha = 0.0
            self._partition = alpha
        else:
            # When not log-linear, partition is a scale.
            if not alpha:
                alpha = 1.0
            self._partition = 1.0 / alpha
        self._alpha = alpha
        self._factors = factors
        self._is_energy_funcs = is_energy_funcs

    def __repr__(self):
        prefix = "Network("
        base_indent = '\n' + ' ' * len(prefix)
        factor_indent = base_indent + ' '
        factors_repr = '[{}]'.format(
            (',' + factor_indent).join(factor_indent.join(
                repr(fac).split('\n')) for fac in self.factors))
        return "{}{})".format(
            prefix,
            (',' + base_indent).join(
                [factors_repr,
                 "names_order={}".format(self.names),
                 "alpha={}".format(self.alpha),
                 "is_energy_funcs={}".format(self.is_energy_funcs)]))

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
        return self._is_energy_funcs

    def partition(self, alpha=None):
        '''
        Apply a partitioning constant alpha to the network. The current
        partitioning constant remains intact so that the network returned has a
        sum over its joint distribution scaled by (1. / alpha).

        N. B. For log-linear models, alpha is assumed to be in log space.

        When alpha=None, compute the partition overall network states. This can
        be intractable for large networks.
        '''
        if alpha:
            if self.is_energy_funcs:
                return Network(self.factors,
                               alpha=self.alpha + alpha,
                               names_order=self.names,
                               is_energy_funcs=self.is_energy_funcs)
            else:
                return Network(self.factors,
                               alpha=self.alpha * alpha,
                               names_order=self.names,
                               is_energy_funcs=self.is_energy_funcs)
        else:
            state_sets = [range(n) for n in self.nstates]
            if self.is_energy_funcs:
                network_no_scale = Network(self.factors,
                                           alpha=0,
                                           names_order=self.names,
                                           is_energy_funcs=self.is_energy_funcs)
                alpha = np.log(np.sum(np.exp(
                    [network_no_scale(assignment)
                     for assignment in it.product(*state_sets)])))
            else:
                network_no_scale = Network(self.factors,
                                           alpha=1,
                                           names_order=self.names,
                                           is_energy_funcs=self.is_energy_funcs)
                alpha = np.sum(
                    [network_no_scale(assignment)
                     for assignment in it.product(*state_sets)])
            return Network(self.factors,
                           alpha=alpha,
                           names_order=self.names,
                           is_energy_funcs=self.is_energy_funcs)

    def to_energy_funcs(self):
        '''
        Convert potentials to energy functions -ln(phi).
        '''

        if self.is_energy_funcs:
            return Network(self.factors, alpha=self.alpha,
                           names_order=self.names, is_energy_funcs=True)
        else:
            energy_funcs = [Factor.fromTable(fac.names, np.log(fac.table))
                            for fac in self.factors]
            return Network(energy_funcs, alpha=np.log(self.alpha),
                           names_order=self.names, is_energy_funcs=True)

    def to_linear_funcs(self):
        '''
        Convert potentials to linear functions.
        '''

        if self.is_energy_funcs:
            linear_funcs = [Factor.fromTable(fac.names, np.exp(fac.table))
                            for fac in self.factors]
            return Network(linear_funcs, alpha=np.exp(self.alpha),
                           names_order=self.names, is_energy_funcs=False)
        else:
            return Network(self.factors, alpha=self.alpha,
                           names_order=self.names, is_energy_funcs=False)

    def query(self, *args):
        '''
        Query the network with the given variable states.

        Note that the query can be individual variable states as in
            network(0, 1, ...)
        or a single tuple of states as in
            network((0, 1, ...))
        of a dictionary of states as in
            network({'var1': 0, ...}).
        '''
        if len(args) == 0:
            raise ValueError("Network query cannot be empty.")
        elif len(args) == 1:
            q = args[0]
            # When the query is a single value, wrap it in an iterable type.
            try:
                iter(q)
            except TypeError:
                q = (q,)
        else:
            q = args

        # Convert query to a map if it is not already. The map form of query()
        # simplifies extracting assignments for each factor.
        if not isinstance(q, dict):
            q = dict(p for p in it.izip(self.names, q))

        if self.is_energy_funcs:
            return np.sum((fac(q) for fac in self.factors)) - self._partition
        else:
            return np.prod([fac(q) for fac in self.factors]) * self._partition


def network_to_dag(network):
    ''' Convert an mrf Network to a networkx DiGraph. '''

    edges = [(c, f.names[-1]) for f in network.factors for c in f.names[:-1]]
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def network_to_mrf(network):
    ''' Convert an mrf Network to a networkx Graph. '''

    edges = [(c, f.names[-1]) for f in network.factors for c in f.names[:-1]]
    g = nx.Graph()
    g.add_edges_from(edges)
    return g
