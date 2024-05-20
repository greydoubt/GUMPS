# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Test module for the Ackley kernel."

import unittest

import numpy as np
import pandas as pd

from gumps.kernels.ackley_kernel import (AckleyFirstTerm, AckleyFunction,
                                         AckleySecondTerm, AckleyFirstTermState,
                                         AckleyFunctionState, AckleySecondTermState)
from gumps.studies import SimpleSimulationStudy

class TestAckleyKernel(unittest.TestCase):
    "test the ackley kernel"

    def test_ackley_first_term_get_state_class(self):
        "test the ackley first term kernel"
        kernel = AckleyFirstTerm()
        self.assertEqual(kernel.get_state_class(), AckleyFirstTermState)

    def test_ackley_first_term_user_defined_function(self):
        "test the ackley first term with a user defined function"
        problem = {'x' : np.array([1,2]),'a' : 1,'b' : 2 ,'term1' : None}
        kernel = AckleyFirstTerm()

        states = kernel.f(problem)

        mean = np.mean(problem['x']**2)
        term1 = (-problem['a']) * np.exp( - problem['b'] * np.sqrt(mean))
        self.assertAlmostEqual(states.term1, term1)

    def test_ackley_first_term_study(self):
        "test the ackley first term with a study"
        problem = {'x' : np.array([1,2]),'a' : 1,'b' : 2 ,'term1' : None}
        kernel = AckleyFirstTerm()
        study = SimpleSimulationStudy(problem=problem, kernel=kernel)
        study.run()

        mean = np.mean(problem['x']**2)
        term1 = (-problem['a']) * np.exp( - problem['b'] * np.sqrt(mean))

        self.assertAlmostEqual(study.states[0].term1, term1)

    def test_ackley_second_term_get_state_class(self):
        "test the ackley second term kernel"
        kernel = AckleySecondTerm()
        self.assertEqual(kernel.get_state_class(), AckleySecondTermState)

    def test_ackley_second_term_user_defined_function(self):
        "test the ackley second term with a user defined function"
        problem = {'x' : np.array([1,2]), 'c' : 1, 'term2' : None}
        kernel = AckleySecondTerm()

        states = kernel.f(problem)

        mean = np.mean(np.cos(problem['c']*problem['x']))
        term2 = - np.exp(mean)
        self.assertAlmostEqual(states.term2, term2)

    def test_ackley_second_term_study(self):
        "test the ackley second term with a study"
        problem = {'x' : np.array([1,2]), 'c' : 1, 'term2' : None}
        kernel = AckleySecondTerm()
        study = SimpleSimulationStudy(problem=problem, kernel=kernel)
        study.run()

        mean = np.mean(np.cos(problem['c']*problem['x']))
        term2 = - np.exp(mean)

        self.assertAlmostEqual(study.states[0].term2, term2)

    def test_ackley_function_get_state_class(self):
        "test the ackley function kernel"
        kernel = AckleyFunction()
        self.assertEqual(kernel.get_state_class(), AckleyFunctionState)

    def test_ackley_function_user_defined_function(self):
        "test the ackley function with a user defined function"
        problem = {'term1' : 1 ,'a' : 1,'term2' : 1}
        kernel = AckleyFunction()

        states = kernel.f(problem)

        total = problem['term1'] + problem['term2'] + problem['a'] + np.exp(1)

        self.assertAlmostEqual(states.total, total)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAckleyKernel)
    unittest.TextTestRunner(verbosity=2).run(suite)
