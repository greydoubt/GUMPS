# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import numpy as np
import pandas as pd

from gumps.kernels.ackley_complete_kernel import AckleyCompleteKernel, AckleyCompleteState
from gumps.studies import SimpleSimulationStudy


class TestAckleyCompleteKernel(unittest.TestCase):
    "Test the ackley complete kernel"

    def test_get_state_class(self):
        "Test the complete ackley kernel"
        kernel = AckleyCompleteKernel()
        self.assertEqual(kernel.get_state_class(), AckleyCompleteState)

    def test_user_defined_function(self):
        "test the complete ackley fucntion with user defined function"
        problem = {'x' : np.array([1,2]),'a' : 1,'b' : 2 ,'term1' : None, 'c' : 1, 'term2' : None, 'total':None}
        kernel = AckleyCompleteKernel()

        states = kernel.f(problem)

        sum_squared = np.sum(problem['x']**2)
        term1 = (-problem['a']) * np.exp( - problem['b'] * np.sqrt(sum_squared/len(problem['x'])))
        mean = np.mean(np.cos(problem['c']*problem['x']))
        term2 = - np.exp(mean)
        total = term1 + term2 + problem['a'] + np.exp(1)
        self.assertAlmostEqual(states.total,total)

    def test_ackley_kernel_study(self):
        "test the ackley complete kernel in a graph kernel with simple study"
        kernel = AckleyCompleteKernel()
        
        problem = {'x' : np.array([1,2]),'a' : 1,'b' : 2 ,'c' : 1}
       
        study = SimpleSimulationStudy(problem=problem, kernel=kernel)
        study.run()
        result = study.state_frame()

        sum_squared = np.sum(problem['x']**2)
        term1 = (-problem['a']) * np.exp( - problem['b'] * np.sqrt(sum_squared/len(problem['x'])))
        mean = np.mean(np.cos(problem['c']*problem['x']))
        term2 = - np.exp(mean)
        total = term1 + term2 + problem['a'] + np.exp(1)

        correct = {'term1' : term1, 'term2' : term2, 'total': total }

        pd.testing.assert_frame_equal(result[['term1', 'term2', 'total']],
                                      pd.DataFrame(correct, index=[0]))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAckleyCompleteKernel)
    unittest.TextTestRunner(verbosity=2).run(suite)