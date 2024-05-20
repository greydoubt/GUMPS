# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import pandas as pd
import shutil
import tempfile
import json

from gumps.common import population

from pathlib import Path

class TestPopulation(unittest.TestCase):
    "test the population class"

    def test_convert_dict_success(self):
        "test a dictionary is converted to a pd.Series correctly"
        test = {'a':1, 'b': 2.5}

        series = population.convert_dict(test)

        correct = pd.Series({'a':1.0, 'b': 2.5})

        pd.testing.assert_series_equal(series, correct)

    def test_convert_dict_fail_str(self):
        "test a diciontary with incorrect input to make sure it doesn't convert"
        with self.assertRaises(ValueError):
            population.convert_dict({'a':"test"})

    def test_convert_dict_fail_str2(self):
        "test a diciontary with incorrect input to make sure it doesn't convert"
        with self.assertRaises(ValueError):
            population.convert_dict("test")

    def test_convert_dict_fail_list(self):
        "test a diciontary with incorrect input to make sure it doesn't convert"
        with self.assertRaises(TypeError):
            population.convert_dict([1,2,3])

    def test_convert_dict_pass_series(self):
        "test a diciontary with incorrect input to make sure it doesn't convert"
        correct = pd.Series({'a':1.0, 'b': 2.5})

        series = population.convert_dict(correct)

        pd.testing.assert_series_equal(series, correct)

    def test_create_loss(self):
        "create a valid loss object"
        loss = population.Loss({'a':1.0, 'b': 2.5})

        self.assertTrue(loss.valid)

    def test_create_loss_invalid(self):
        "create an invalid loss object"
        loss = population.Loss({})

        self.assertFalse(loss.valid)

    def test_create_loss_invalid_none(self):
        "create an invalid loss object"
        loss = population.Loss(None)

        self.assertFalse(loss.valid)

    def test_del_loss(self):
        "create a valid loss object"
        loss = population.Loss({'a':1.0, 'b': 2.5})

        self.assertTrue(loss.valid)

        del loss.values

        self.assertFalse(loss.valid)

    def test_set_loss(self):
        "create a valid loss object"
        loss = population.Loss(None)

        self.assertFalse(loss.valid)

        loss.values = {'a':1.0, 'b': 2.5}

        self.assertTrue(loss.valid)

    def test_create_individual_false(self):
        "check that no exception is made for correct creation"
        ind = population.Individual(parameters={'a':1.0, 'b': 2.5})

        self.assertFalse(ind.valid)

    def test_create_individual_true(self):
        "check that no exception is made for correct creation"
        ind = population.Individual(parameters={'a':1.0, 'b': 2.5},
                loss={'a':1.0, 'b': 2.5})

        self.assertTrue(ind.valid)

    def test_save_name_base(self):
        "make sure the hash is created correctly"
        ind = population.Individual(parameters={'a':1.0, 'b': 2.5})
        save_name = ind.save_name_base

        self.assertEqual(save_name, "9eb5452fa5d21ac7136bf18f8c86e312")


class TestPopulationIO(unittest.TestCase):
    "test the population class"

    def setUp(self):
        "create a temporary directory for results"
        self.results_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        "remove temporary directory"
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)

    def test_save_path(self):
        "save the data to json and read it back"
        ind = population.Individual(parameters={'a':1.0, 'b': 2.5})

        path = self.results_dir / 'test.json'

        ind.save_full_path(path)

        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        self.assertDictEqual(ind.loss.values.to_dict(), data['loss'])
        self.assertDictEqual(ind.parameters.to_dict(), data['parameters'])
        self.assertDictEqual(ind.transformed_parameters.to_dict(), data['transformed_parameters'])


    def test_save(self):
        "save the data to json and read it back"
        ind = population.Individual(parameters={'a':1.0, 'b': 2.5})

        path = self.results_dir / f"{ind.save_name_base}.json"

        ind.save(self.results_dir)

        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        self.assertDictEqual(ind.loss.values.to_dict(), data['loss'])
        self.assertDictEqual(ind.parameters.to_dict(), data['parameters'])
        self.assertDictEqual(ind.transformed_parameters.to_dict(), data['transformed_parameters'])

class TestMeta(unittest.TestCase):
    "test the IndividualMeta class"

    def test_loss(self):
        "test the get loss function"
        parameters = pd.Series({'a':1.0, 'b': 2.5})
        loss = pd.Series({'a':1.0, 'b': 2.5})
        transformed_parameters = pd.Series({'a':1.0, 'b': 2.5})
        loss_full = pd.Series({'a':1.0, 'b': 2.5})
        loss_raw = pd.Series({'a':1.0, 'b': 2.5})
        loss_absolute = pd.Series({'a':1.0, 'b': 2.5})

        ind = population.IndividualMeta(loss=loss,
            parameters=parameters,
            transformed_parameters=transformed_parameters)
        ind.loss_full=loss_full
        ind.loss_raw=loss_raw
        ind.loss_absolute=loss_absolute

        loss_result = ind.get_loss()

        correct = {'loss':loss_full.to_dict(), 'loss_raw':loss_raw.to_dict(),
            'loss_absolute':loss_absolute.to_dict()}

        self.assertDictEqual(loss_result, correct)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPopulation)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestPopulationIO)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestMeta)
    unittest.TextTestRunner(verbosity=2).run(suite)
