'''
Tests for ve module.
'''
import itertools as its
import unittest

import numpy as np

import mrf


class TestVe(unittest.TestCase):
    ''' Tests for variable elimination. '''

    def simple_model(self):

        variables = ['0', '1', '2', '3', '4']
        factors = ((['0', '1'], np.array([[1.0, 2.0],
                                          [3.0, 4.0],
                                          [5.0, 6.0]])),
                   (['1', '2', '3'], np.array([[[1.0, 2.0],
                                                [3.0, 4.0],
                                                [5.0, 6.0]],
                                               [[1.3, 2.3],
                                                [3.3, 4.3],
                                                [5.3, 6.3]]])),
                   (['1'], np.array([0.1, 0.9])),
                   (['1', '4'], np.array([[0.2, 0.3],
                                          [0.4, 0.5]])))
        return mrf.Network([
            mrf.Factor(my_variables, data)
            for my_variables, data in factors],
            names_order=variables, is_energy_funcs=False).partition()

    def cmb_partition_gen(self, model):
        '''
        Split model into all combinations of partitions of size [1, len - 1] and
        all values of is_energy_funcs.

        :return tuple[mrf.Netork, list, list]: The input model as either energy
        functions or linear functions, and all possible combinations of
        variables of length 1 to (len(network.names) - 1) with the remaining
        variables in the final list.
        '''
        for is_energy_funcs in [True, False]:
            for lt_len in range(2, len(model.names)):
                for left_partition in its.combinations(model.names, lt_len):
                    right_partition = tuple(var for var in model.names
                                            if var not in left_partition)
                    if is_energy_funcs:
                        yield model.to_energy_funcs(), \
                            list(left_partition), list(right_partition)
                    else:
                        yield model.to_linear_funcs(), \
                            list(left_partition), list(right_partition)

    def test_eliminate(self):

        model = self.simple_model()
        factors_copy = [(list(factor.names), np.copy(factor.table))
                        for factor in model.factors]
        var_to_nstates = dict((var, nstates)
                              for var, nstates in
                              zip(model.names, model.nstates))

        # Test every possible elimination in both log-linear and linear space.
        for test_model, elim_vars, elim_names in self.cmb_partition_gen(model):

            elim_model = mrf.eliminate(test_model, elim_vars).to_linear_funcs()
            elim_nstates = [var_to_nstates[var] for var in elim_names]
            table_expected = np.zeros(elim_nstates)

            self.assertEqual(elim_names, elim_model.names)
            self.assertEqual(elim_nstates, elim_model.nstates)

            model_alpha = 0
            for perm in its.product(*[range(n) for n in model.nstates]):
                perm_dict = dict((var, x) for var, x in zip(model.names, perm))
                expected_perm = tuple(perm_dict[var] for var in elim_names)
                table_expected[expected_perm] += model(perm)
                model_alpha += model(perm)

            # Model sums to 1.
            self.assertAlmostEqual(1.0, model_alpha)

            elim_model_alpha = 0
            for perm in its.product(*[range(n) for n in elim_model.nstates]):
                expected = table_expected[perm]
                actual = elim_model(perm)
                self.assertAlmostEqual(expected, actual)
                elim_model_alpha += actual

            # Eliminated model sums to 1.
            self.assertAlmostEqual(1.0, elim_model_alpha)

            # Original model is unchanged.
            self.assertEqual(len(factors_copy), len(model.factors))
            for (names, table), factor in zip(factors_copy, model.factors):
                self.assertEqual(names, factor.names)
                np.testing.assert_array_equal(table, factor.table)

    def test_condition_eliminate_eq_eliminate(self):

        model = self.simple_model()
        factors_copy = [(list(factor.names), np.copy(factor.table))
                        for factor in model.factors]

        # Test that condition_eliminate() without any conditioning is equal to
        # elimination only.
        for test_model, elim_vars, elim_names in self.cmb_partition_gen(model):

            elim_model = mrf.eliminate(
                test_model, elim_vars).to_linear_funcs()
            cond_elim_model = mrf.condition_eliminate(
                test_model, elim_model.names, {}, elim_vars).to_linear_funcs()

            self.assertEqual(elim_model.names, cond_elim_model.names)
            self.assertEqual(elim_model.nstates, cond_elim_model.nstates)

            for perm in its.product(*[range(n) for n in elim_model.nstates]):
                self.assertEqual(elim_model(perm), cond_elim_model(perm))

            # Original model is unchanged.
            self.assertEqual(len(factors_copy), len(model.factors))
            for (names, table), factor in zip(factors_copy, model.factors):
                self.assertEqual(names, factor.names)
                np.testing.assert_array_equal(table, factor.table)

    def test_condition_eliminate(self):

        model = self.simple_model()
        factors_copy = [(list(factor.names), np.copy(factor.table))
                        for factor in model.factors]
        var_to_nstates = dict((var, nstates)
                              for var, nstates in
                              zip(model.names, model.nstates))

        def parition_scope_gen(non_cond_names):
            for num_scope in range(1, len(non_cond_names) + 1):
                for scope in its.combinations(non_cond_names, num_scope):
                    elim_vars = [var for var in non_cond_names
                                 if var not in scope]
                    yield list(scope), elim_vars

        # Test elimination with conditioning.
        for test_model, cond_names, non_cond_names in \
                self.cmb_partition_gen(model):

            # Partition non_cond_names into elim_vars and scope.
            for scope, elim_vars in parition_scope_gen(non_cond_names):
                cond_nstates = [var_to_nstates[var] for var in cond_names]

                for cond_perm in its.product(*[range(n) for n in cond_nstates]):
                    evidence = dict((var, state)
                                    for var, state in zip(cond_names,
                                                          cond_perm))

                    cond_elim_model_norm = mrf.condition_eliminate(
                        test_model, scope,
                        evidence, model.names, normalize=True).to_linear_funcs()
                    cond_elim_model = mrf.condition_eliminate(
                        test_model, scope,
                        evidence, model.names).to_linear_funcs().partition()

                    scope_nstates = [var_to_nstates[var] for var in scope]
                    table_expected = np.zeros(scope_nstates)

                    # Loop over assignments to the original model with evidence
                    # held constant.
                    ranges = [range(var_to_nstates[var]) if var not in evidence
                              else [evidence[var]] for var in model.names]
                    for perm in its.product(*ranges):
                        perm_dict = dict((var, x) for var, x in
                                         zip(model.names, perm))
                        scope_perm = tuple(perm_dict[var] for var in scope)
                        table_expected[scope_perm] += model(perm)

                    table_expected /= np.sum(table_expected)

                    # Assert expected equals actual condition eliminated model.
                    for perm in its.product(*[range(n) for n in
                                              cond_elim_model.nstates]):
                        expected = table_expected[perm]
                        actual = cond_elim_model(perm)
                        actual_norm = cond_elim_model_norm(perm)
                        self.assertAlmostEqual(expected, actual)
                        self.assertAlmostEqual(expected, actual_norm)

                    # Original model is unchanged.
                    self.assertEqual(len(factors_copy), len(model.factors))
                    for (names, table), factor in zip(factors_copy,
                                                      model.factors):
                        self.assertEqual(names, factor.names)
                        np.testing.assert_array_equal(table, factor.table)


if __name__ == '__main__':
    unittest.main()
