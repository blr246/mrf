'''
Tests for mrf module.
'''
import math
import itertools as its
import unittest

import numpy as np

import mrf


class TestMrf(unittest.TestCase):

    def test_factor_properties(self):
        """ Test simple factor properties. """

        variables = ['0', '1']
        data = np.array([[1.0, math.e],
                         [1.0, math.e],
                         [math.e, 1.0]])
        factor = mrf.Factor(variables, data)

        self.assertTrue('0' in factor and factor.contains_var('0'))
        self.assertTrue('1' in factor and factor.contains_var('1'))
        self.assertFalse('something' in factor
                         or factor.contains_var('something'))
        self.assertEqual(variables, factor.names)
        self.assertEqual((3, 2), factor.nstates)
        self.assertTrue(data is factor.table)

        self.assertEqual(1.0, factor((0, 0)))
        self.assertEqual(1.0, factor.query((0, 0)))
        self.assertEqual(math.e, factor((0, 1)))
        self.assertEqual(math.e, factor.query((0, 1)))

        self.assertEqual(1.0, factor(0, 0))
        self.assertEqual(1.0, factor.query(0, 0))
        self.assertEqual(math.e, factor(0, 1))
        self.assertEqual(math.e, factor.query(0, 1))

        self.assertEqual(1.0, factor({'0': 1, '1': 0}))
        self.assertEqual(1.0, factor.query({'0': 1, '1': 0}))
        self.assertEqual(math.e, factor({'0': 2, '1': 0}))
        self.assertEqual(math.e, factor.query({'0': 2, '1': 0}))

        factor_evid = factor.given_evidence({'0': 0})
        self.assertEqual(1.0, factor_evid(0))
        self.assertEqual(math.e, factor_evid(1))
        self.assertRaises(ValueError, factor.given_evidence, {'0': 0, '1': 0})

    def test_network_properties(self):
        """ Test simple network properties. """

        # Create network with two cliques and parameter values of either 1 or e
        # such that P(0=?, 1=0, 2=?, 3=?) = P(0=?, 1=1, 2=?, 3=?).
        variables = ['0', '1', '2', '3']
        nstates = [3, 2, 3, 2]
        factors = ((['0', '1'], np.array([[1.0, math.e],
                                          [1.0, math.e],
                                          [1.0, math.e]])),
                   (['1', '2', '3'], np.array([[[math.e, math.e],
                                                [math.e, math.e],
                                                [math.e, math.e]],
                                               [[1.0, 1.0],
                                                [1.0, 1.0],
                                                [1.0, 1.0]]])))
        model = mrf.Network([
            mrf.Factor(my_variables, data)
            for my_variables, data in factors],
            names_order=variables, is_energy_funcs=False)

        self.assertEqual(variables, model.names)
        self.assertEqual(nstates, model.nstates)
        self.assertEqual(len(factors), len(model.factors))
        self.assertEqual(1.0, model.alpha)
        self.assertEqual(False, model.is_energy_funcs)

        # Convert to energy functions and observe that every setting of the
        # network equals 1.0 (and that the linear model's response is math.e).
        # Also convert to linear from energy and test queries.
        model_energy = model.to_energy_funcs()
        model_energy_again = model_energy.to_energy_funcs()
        model_energy_linear = model_energy.to_linear_funcs()
        model_linear_again = model.to_linear_funcs()
        for assignment in its.product(*[range(val) for val in nstates]):
            self.assertEqual(math.e, model(*assignment))
            self.assertEqual(1.0, model_energy(*assignment))
            self.assertEqual(1.0, model_energy_again(*assignment))
            self.assertEqual(math.e, model_energy_linear(*assignment))
            self.assertEqual(math.e, model_linear_again(*assignment))

            # Also test other query APIs.
            for test_model, expected in [(model, math.e),
                                         (model_energy, 1.0),
                                         (model_energy_again, 1.0),
                                         (model_energy_linear, math.e),
                                         (model_linear_again, math.e)]:
                self.assertEqual(expected, test_model.query(*assignment))
                self.assertEqual(expected, test_model(assignment))
                self.assertEqual(expected, test_model.query(assignment))
                assignment_dict = dict(zip(variables, assignment))
                self.assertEqual(expected, test_model(assignment_dict))
                self.assertEqual(expected, test_model.query(assignment_dict))

        # Partition the network by doubling alpha.
        model_half = model.partition(2.0)
        self.assertEqual(model(0, 0, 0, 0) / 2.,
                         model_half(0, 0, 0, 0))
        model_half_third = model_half.partition(3.0)
        self.assertEqual(model(0, 0, 0, 0) / (2. * 3.),
                         model_half_third(0, 0, 0, 0))
        ln_2 = math.log(2)
        model_energy_half = model_energy.partition(ln_2)
        self.assertEqual(model_energy(0, 0, 0, 0) - ln_2,
                         model_energy_half(0, 0, 0, 0))
        ln_3 = math.log(3)
        model_energy_half_third = model_energy_half.partition(ln_3)
        self.assertEqual(model_energy(0, 0, 0, 0) - (ln_2 + ln_3),
                         model_energy_half_third(0, 0, 0, 0))

        # Partitioning should make network states sum to 1, even when the model
        # has a non-trivial alpha to begin.
        model_norm = model.partition(10).partition()
        z = np.sum([model_norm(x)
                    for x in its.product(*[range(n)
                                           for n in model_norm.nstates])])
        self.assertAlmostEqual(1.0, z)

    def _test_nfactor_helper(self, is_energy_funcs, largest_clique, nstates):
        """
        Test multi-factor networks. The energy function parameters controls how
        factor states are aggregated into the overall network state (mul or
        add).

        :param bool is_energy_funcs: factors are either energy functions
        (log-linear) or not
        :param int largest_clique: tests a network of factors [2, 3, ...,
        largest_factor]
        :param int nstates: number of states for variables
        """

        energy_op = np.sum if is_energy_funcs else np.product
        num_cliques = largest_clique - 1
        clique_sizes = range(2, largest_clique + 1)
        clique_data = [np
                       .fromiter(xrange(1, (nstates**clique_size) + 1),
                                 dtype=float)
                       .reshape(tuple([nstates] * clique_size))
                       for clique_size in clique_sizes]

        # Each larger clique must contain at least 2 variables not contained
        # in the smaller cliques since otherwise the smaller clique would be
        # contained within the larger clique and not a proper factor.
        num_variables = 2 * num_cliques
        variables = [str(idx) for idx in xrange(num_variables)]
        factors = [mrf.Factor(variables[idx:idx + clique_size], data)
                   for (idx, clique_size), data
                   in its.izip(enumerate(clique_sizes), clique_data)]
        model = mrf.Network(factors,
                            names_order=variables,
                            is_energy_funcs=is_energy_funcs)

        # Compute expected energy for every possible variable assignment.
        for assignment in its.product(range(nstates), repeat=num_variables):
            # Get factor state for each clique.
            energies = [data[assignment[idx:idx + clique_size]]
                        for (idx, clique_size), data
                        in its.izip(enumerate(clique_sizes), clique_data)]
            query_expected = energy_op(energies)
            query_actual = model(*assignment)
            self.assertAlmostEqual(query_expected, query_actual)

    def test_nfactor_linear(self):
        """ Test multi-factor network that is not log-linear. """
        for largest_clique, nstates in its.product(xrange(2, 6), xrange(2, 4)):
            self._test_nfactor_helper(False, largest_clique, nstates)

    def test_nfactor_log_linear(self):
        """ Test multi-factor network that is log-linear. """
        for largest_clique, nstates in its.product(xrange(2, 6), xrange(2, 4)):
            self._test_nfactor_helper(True, largest_clique, nstates)


if __name__ == '__main__':
    unittest.main()
