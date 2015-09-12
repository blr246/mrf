import itertools as its
import numpy as np

import mrf
import ve


def load(file, is_energy_funcs=False):
    '''
    Parse the UAI network data to a mrf.Network of Factor instances.

    Parameters
    ----------
    file : iterable of string
        Iterator over lines of the UAI file. Can be file, list, etc.
    Returns
    -------
    network : mrf.Network
        The network stored in the UAI file.
    '''

    def burn_to_next_line(lineiter):
        while True:
            line = lineiter.next().strip()
            if len(line) > 0:
                return line

    lineiter = iter(file)

    # Read network type.
    net_type = lineiter.next().strip().lower()
    if not net_type == 'markov':
        raise ValueError("I only read MARKOV networks")

    # Read number of variables.
    num_vars = int(lineiter.next().strip())

    # Read cardinality.
    cardinality = [int(val) for val in lineiter.next().strip().split()]

    # Read num cliques.
    num_cliques = int(lineiter.next().strip())

    # Read variables within each clique.
    clique_descs = [tuple([int(val) for val in lineiter.next().strip().split()])
                    for _ in range(num_cliques)]

    # Read factor tables.
    factors = []
    for clique_desc in clique_descs:

        # Read num entries and get dims for parameter table.
        num_entries = int(burn_to_next_line(lineiter))
        pdims = [cardinality[var] for var in clique_desc[1:]]
        assert(num_entries == np.product(pdims))

        # Read factor parameters.
        data = [float(n) for n in lineiter.next().strip().split()]
        while len(data) < num_entries:
            data.extend([float(n) for n in lineiter.next().strip().split()])
        assert(num_entries == len(data))

        # Load into a table.
        table = np.array(data).reshape(pdims)

        # Make a new factor.
        factors.append(mrf.Factor([str(i) for i in clique_desc[1:]], table))

    # Construct a newtork.
    return mrf.Network(
        factors, names_order=[str(i) for i in range(num_vars)],
        is_energy_funcs=is_energy_funcs)


def dump(network, file):
    '''
    Write the network in UAI format to the given file.

    Parameters
    ----------
    network : mrf.Network
        The network to serialize.
    file : file
        File for UAI output.
    Returns
    -------
    None
    '''

    # 1. Network type.
    file.write('MARKOV\n')

    # 2. Number of variables.
    file.write('{}\n'.format(len(network.names)))

    # 3. Variable cardinalities.
    file.write('{}\n'.format(' '.join([str(val) for val in network.nstates])))

    # 4. Number of cliques.
    file.write('{}\n'.format(len(network.factors)))

    # 5. [CLIQUE_VARS I1 ... IN]
    name_to_idx = dict(its.izip(network.names, its.count()))
    file.writelines(['{} {}\n'.format(
        len(fac.names), ' '.join([str(name_to_idx[var]) for var in fac.names]))
        for fac in network.factors])

    # 6. [n*m
    #      x11 ... x1m
    #       .  .    .
    #       .   .   .
    #       .    .  .
    #      xn1 ... xnm]
    for fac in network.factors:
        file.write('\n')
        file.write('{}\n'.format(fac.table.size))
        if len(fac.names) > 1:
            file.writelines([' {}\n'.format(
                ' '.join(['{:.8f}'.format(param) for param in row]))
                for row in fac.table])
        else:
            file.write(' {}\n'.format(
                ' '.join(['{:.8f}'.format(param) for param in fac.table])))

    # Final newline.
    file.write('\n')


def with_evidence(network, evidence_file):
    '''
    Condition network using the given evidence file.

    :return tuple[mrf.Network, dict]: network conditioned on evidence and a dict
    of the evidence variables and their values
    '''

    data = evidence_file.read().replace('\n', '').strip().split()
    num_observed = int(data[0])
    if len(data) != (2 * num_observed) + 1:
        raise ValueError("Evidence file is malformed; expecting\n"
                         "NUM_OBSERVED VAR1_NAME VAR1_VALUE ...")
    if num_observed > 0:
        evidence = {var: int(val)
                    for var, val in its.izip(data[1::2], data[2::2])}
        observed = set(network.names) - set(evidence.keys())
        return ve.condition_eliminate(network, observed, evidence, []), evidence
    else:
        return network, {}
