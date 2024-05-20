# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Test the model interface specification"

import unittest

import attrs
import pandas as pd
import pint

from gumps.interface import model_interface as interface
from gumps.interface import model_units as units


@attrs.define
class TestUnit(units.Mass):
    @classmethod
    def bounds_units(cls):
        return "kg"

    @classmethod
    def lower_bound(cls) -> float|int:
        return 0

    @classmethod
    def upper_bound(cls) -> float|int:
        return 10


@attrs.define
class TestUnit2(units.Mass):
    @classmethod
    def bounds_units(cls):
        return "s"

    @classmethod
    def lower_bound(cls) -> float|int:
        return 0

    @classmethod
    def upper_bound(cls) -> float|int:
        return 10


@attrs.define
class TestUnits(interface.Units):
    @staticmethod
    def units_used() -> dict[str, interface.Unit]:
        return {'a': TestUnit,
                'b': TestUnit,
                'c': TestUnit2}


@attrs.define
class TestCollection(interface.UnitCollection):
    a: TestUnit
    b: TestUnit
    c: TestUnit2


class TestModelInterfaceUnit(unittest.TestCase):
    "test the model interface specification"

    def test_validate_units(self):
        "test the mass unit"
        TestUnit(1, 'g')


    def test_invalid_units(self):
        "test the mass unit with the wrong units"
        with self.assertRaises(pint.DimensionalityError):
            TestUnit(1, 'm')


    def test_validate_bounds_lower(self):
        "test the mass unit with a value that is too low"
        with self.assertRaises(ValueError):
            TestUnit(-1, 'kg')


    def test_validate_bounds_upper(self):
        "test the mass unit with a value that is too high"
        with self.assertRaises(ValueError):
            TestUnit(11, 'kg')


    def test_validate_bounds_convert(self):
        "test the mass unit with a value that is too high"
        TestUnit(5000, 'g')


    def test_convert(self):
        "test unit conversion"
        self.assertEqual(TestUnit(5000, 'g').convert('kg').value, 5)


class TestModelInterfaceUnits(unittest.TestCase):
    "test the model interface specification for Units"

    def test_validate_units(self):
        "test the mass unit"
        TestUnits(pd.Series({'a':1, 'b':2, 'c':5}), {"a": "g", "b": "kg", "c": "s"})


    def test_invalid_units(self):
        "test the mass unit with the wrong units"
        with self.assertRaises(pint.DimensionalityError):
            TestUnits(pd.Series({'a':1, 'b':2, 'c':5}), {"a": "m", "b": "kg", "c": "s"})


    def test_validate_bounds_lower(self):
        "test the mass unit with a value that is too low"
        with self.assertRaises(ValueError):
            TestUnits(pd.Series({'a':-1, 'b':2, 'c':5}), {"a": "g", "b": "kg", "c":"s"})


    def test_validate_bounds_upper(self):
        "test the mass unit with a value that is too high"
        with self.assertRaises(ValueError):
            TestUnits(pd.Series({'a':1, 'b':11, 'c':5}), {"a": "g", "b": "kg", 'c':'s'})


    def test_validate_bounds_convert(self):
        "test the mass unit with a value that is too high"
        TestUnits(pd.Series({'a':1000, 'b':2000, 'c':5}), {"a": "g", "b": "g", 'c':'s'})


    def test_convert(self):
        "test unit conversion"
        original = TestUnits(pd.Series({'a':1000.0, 'b':2.0, 'c':5}), {"a": "g", "b": "kg", 'c':'s'})
        converted = original.convert({TestUnit: "kg"})
        pd.testing.assert_series_equal(converted.value, pd.Series({'a':1.0, 'b':2.0, 'c':5}))


    def test_convert_dataframe(self):
        "test unit conversion with a dataframe"
        original = TestUnits(pd.DataFrame({'a':[1000.0, 2000.0], 'b':[2.0, 4.0], 'c':[3.0, 5.0]}), {"a": "g", "b": "kg", 'c':'s'})
        converted = original.convert({TestUnit: "kg"})
        pd.testing.assert_frame_equal(converted.value, pd.DataFrame({'a':[1.0, 2.0], 'b':[2.0, 4.0], 'c':[3.0, 5.0]}))

    def test_getitem(self):
        "test that getitem works"
        original = TestUnits(pd.DataFrame({'a':[1000.0, 2000.0], 'b':[2.0, 4.0], 'c':[3.0, 5.0]}), {"a": "g", "b": "kg", 'c':'s'})

        series = TestUnits(pd.Series({'a':1000.0, 'b':2.0, 'c':3.0}), {"a": "g", "b": "kg", 'c':'s'})

        selected = original[0]

        pd.testing.assert_series_equal(selected.value, series.value)
        self.assertEqual(selected.units, series.units)


    def test_getitem_typeerror(self):
        "test that a typeerror is raised if the value is not a DataFrame"
        original = TestUnits(pd.Series({'a':1000.0, 'b':2.0, 'c':5.0}), {"a": "g", "b": "kg", 'c':'s'})
        with self.assertRaises(TypeError):
            original[0]


    def test_indexerror(self):
        "test that an indexerror is raised if the index is out of range"
        original = TestUnits(pd.DataFrame({'a':[1000.0, 2000.0], 'b':[2.0, 4.0], 'c':[3.0, 5.0]}), {"a": "g", "b": "kg", 'c':'s'})

        with self.assertRaises(IndexError):
            original[2]


class TestModelInterfaceCollection(unittest.TestCase):

    def test_convert(self):
        "convert a collection of units"
        data = TestCollection(TestUnit(1, 'kg'), TestUnit(2, 'g'), TestUnit2(5, 's'))
        converted = data.convert({TestUnit: "g"})
        self.assertEqual(converted.a.value, 1000)
        self.assertEqual(converted.b.value, 2)


    def test_get_series(self):
        "get a series from a collection of units"
        data = TestCollection(TestUnit(1, 'kg'), TestUnit(2, 'g'), TestUnit2(5, 's'))
        series = data.get_series()
        pd.testing.assert_series_equal(series, pd.Series({'a':1, 'b':2, 'c':5}))


    def test_get_units(self):
        "get the units from a collection of units"
        data = TestCollection(TestUnit(1, 'kg'), TestUnit(2, 'g'), TestUnit2(5, 's'))
        series = data.get_units()
        self.assertDictEqual(series, {'a':pint.Quantity(1, 'kg'), 'b':pint.Quantity(1, 'g'), 'c':pint.Quantity(1, 's')})


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelInterfaceUnit)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelInterfaceUnits)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelInterfaceCollection)
    unittest.TextTestRunner(verbosity=2).run(suite)
