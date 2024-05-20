# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import json

import gumps.common.pareto as pareto
import gumps.common.population as pop

class TestPareto(unittest.TestCase):
    "test the pareto class"

    def test_dominates_succeed(self):
        "test that a dominates b"
        a = pop.Loss({'a':1, 'b':2})
        b = pop.Loss({'a':1, 'b':2.1})

        self.assertTrue(pareto.dominates(a,b))

    def test_dominates_fail(self):
        "test that a doesn't dominate b"
        a = pop.Loss({'a':1, 'b':2})
        b = pop.Loss({'a':0.9, 'b':2})

        self.assertFalse(pareto.dominates(a,b))

    def test_similar_success(self):
        "test that a and b are similar"
        a = pop.Loss({'a':1, 'b':2})
        b = pop.Loss({'a':1.001, 'b':2.001})

        self.assertTrue(pareto.similar_func(a.values,b.values))

    def test_similar_fail(self):
        "test that a and b are similar"
        a = pop.Loss({'a':1, 'b':2})
        b = pop.Loss({'a':1.2, 'b':2.01})

        self.assertFalse(pareto.similar_func(a.values,b.values))

    def test_update_true(self):
        "update the best with a better item"
        atol = 1e-8
        rtol = 1e-2

        best = pd.Series({'y1':1, 'y2':1})

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})
        ind_b = pop.Individual({'x1':2, 'x2':1}, loss={'y1':2, 'y2':1})

        best_items = {'y1':ind_a, 'y2':ind_b}

        ind = pop.Individual({'x1':1.2, 'x2':2}, loss={'y1':0.9, 'y2':2})

        success = pareto.update_best(ind, best, best_items, atol, rtol)

        self.assertTrue(success)


    def test_update_false(self):
        "try to add a new item that is worse"
        atol = 1e-8
        rtol = 1e-2

        best = pd.Series({'y1':1, 'y2':1})

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})
        ind_b = pop.Individual({'x1':2, 'x2':1}, loss={'y1':2, 'y2':1})

        best_items = {'y1':ind_a, 'y2':ind_b}

        ind = pop.Individual({'x1':1.2, 'x2':2}, loss={'y1':1.1, 'y2':2})

        success = pareto.update_best(ind, best, best_items, atol, rtol)

        self.assertFalse(success)


    def test_update_better_not_significant(self):
        "try to add a new item that is worse"
        atol = 1e-8
        rtol = 1e-1

        best = pd.Series({'y1':1, 'y2':1})

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})
        ind_b = pop.Individual({'x1':2, 'x2':1}, loss={'y1':2, 'y2':1})

        best_items = {'y1':ind_a, 'y2':ind_b}

        ind = pop.Individual({'x1':1.2, 'x2':2}, loss={'y1':0.9, 'y2':2})

        success = pareto.update_best(ind, best, best_items, atol, rtol)

        self.assertFalse(success)

        np.testing.assert_allclose(ind.parameters.to_numpy(), best_items['y1'].parameters.to_numpy())
        np.testing.assert_allclose(ind.loss.values.to_numpy(), best_items['y1'].loss.values.to_numpy())

    def test_dummy_front(self):
        "test that the DummyFront can be created and that the methods work"

        dummy = pareto.DummyFront()

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})
        ind_b = pop.Individual({'x1':2, 'x2':1}, loss={'y1':2, 'y2':1})

        new_members, significant = dummy.update([ind_a, ind_b])

        self.assertFalse(significant)
        self.assertEqual(new_members, [])

    def test_clean_dir(self):
        "test that a directory can be cleaned"

        dummy = pareto.DummyFront()

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)

            with open(directory / "a.json", 'w', encoding='utf-8') as file:
                json.dump({}, file, indent=4, sort_keys=True)

            with open(directory / "b.json", 'w', encoding='utf-8') as file:
                json.dump({}, file, indent=4, sort_keys=True)

            self.assertEqual(len(list(directory.glob("*"))), 2)

            pareto.clean_dir(directory, dummy)

            self.assertEqual(len(list(directory.glob("*"))), 0)

    def test_update_front(self):
        "update the ParetoFront"

        front = pareto.ParetoFront(dimensions=2)

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})
        ind_b = pop.Individual({'x1':2, 'x2':1}, loss={'y1':2, 'y2':1})

        new_members, significant = front.update([ind_a, ind_b])

        self.assertEqual(new_members, [ind_a, ind_b])
        self.assertTrue(significant)

    def test_update_front_func(self):
        "update the ParetoFront"

        front = pareto.ParetoFront(dimensions=2)

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})

        new_members, significant = pareto.update_pareto_front(front, ind_a)

        self.assertTrue(new_members)
        self.assertTrue(significant)

    def test_update_front_fail(self):
        "update the ParetoFront"

        front = pareto.ParetoFront(dimensions=2)

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})
        ind_b = pop.Individual({'x1':2, 'x2':1}, loss={'y1':2, 'y2':1})

        new_members, significant = front.update([ind_a, ind_b])

        ind_c = pop.Individual({'x1':2, 'x2':2}, loss={'y1':2, 'y2':2})

        new_members, significant = front.update([ind_c])

        self.assertFalse(len(new_members))
        self.assertFalse(significant)

    def test_update_front_add_nosig(self):
        "update the ParetoFront"

        front = pareto.ParetoFront(dimensions=2)

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})
        ind_b = pop.Individual({'x1':2, 'x2':1}, loss={'y1':2, 'y2':1})

        new_members, significant = front.update([ind_a, ind_b])

        ind_c = pop.Individual({'x1':2, 'x2':2}, loss={'y1':0.999999, 'y2':2})

        new_members, significant = front.update([ind_c])

        self.assertEqual(new_members, [ind_c])
        self.assertFalse(significant)

    def test_update_entries(self):
        "test the number of entries"
        front = pareto.ParetoFront(dimensions=2)

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})
        ind_b = pop.Individual({'x1':2, 'x2':1}, loss={'y1':2, 'y2':1})

        new_members, significant = front.update([ind_a, ind_b])

        self.assertEqual(front.total_entries(), 2)

    def test_getEntries(self):
        "test the number of entries"
        front = pareto.ParetoFront(dimensions=2)

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})
        ind_b = pop.Individual({'x1':2, 'x2':1}, loss={'y1':2, 'y2':1})

        new_members, significant = front.update([ind_a, ind_b])

        population, fitnesses = front.get_entries()

        correct_population = pd.DataFrame([{'x1':1.0, 'x2':2.0}, {'x1':2.0, 'x2':1.0}])
        correct_fitnesses = pd.DataFrame([{'y1':1.0, 'y2':2.0}, {'y1':2.0, 'y2':1.0}])

        pd.testing.assert_frame_equal(population, correct_population)
        pd.testing.assert_frame_equal(fitnesses, correct_fitnesses)

    def test_update_best(self):
        "test the number of entries"
        front = pareto.ParetoFront(dimensions=2)

        ind_a = pop.Individual({'x1':1, 'x2':2}, loss={'y1':1, 'y2':2})
        ind_b = pop.Individual({'x1':2, 'x2':1}, loss={'y1':2, 'y2':1})

        new_members, significant = front.update([ind_a, ind_b])

        self.assertDictEqual(front.get_best_scores().to_dict(), {'y1':1, 'y2':1})


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPareto)
    unittest.TextTestRunner(verbosity=2).run(suite)
