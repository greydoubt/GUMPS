# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Test the model base units by checking if the units are correct"

import unittest
import pint
from gumps.interface import model_units as units

class TestModelUnits(unittest.TestCase):
    "test the model units"

    def test_mass(self):
        "test the mass unit"
        self.assertEqual(units.Mass.base_units(), "[mass]")
        self.assertTrue(pint.Quantity(1, 'kg').check(units.Mass.base_units()))


    def test_density(self):
        "test the density unit"
        self.assertEqual(units.Density.base_units(), "[mass]/[length]^3")
        self.assertTrue(pint.Quantity(1, 'kg/m^3').check(units.Density.base_units()))


    def test_volume(self):
        "test the volume unit"
        self.assertEqual(units.Volume.base_units(), "[length]^3")
        self.assertTrue(pint.Quantity(1, 'm^3').check(units.Volume.base_units()))


    def test_time(self):
        "test the time unit"
        self.assertEqual(units.Time.base_units(), "[time]")
        self.assertTrue(pint.Quantity(1, 's').check(units.Time.base_units()))


    def test_pressure(self):
        "test the pressure unit"
        self.assertEqual(units.Pressure.base_units(), "[mass]/[length]/[time]^2")
        self.assertTrue(pint.Quantity(1, 'Pa').check(units.Pressure.base_units()))


    def test_temperature(self):
        "test the temperature unit"
        self.assertEqual(units.Temperature.base_units(), "[temperature]")
        self.assertTrue(pint.Quantity(1, 'K').check(units.Temperature.base_units()))


    def test_viscosity(self):
        "test the viscosity unit"
        self.assertEqual(units.Viscosity.base_units(), "[mass]/[length]/[time]")
        self.assertTrue(pint.Quantity(1, 'Pa*s').check(units.Viscosity.base_units()))


    def test_surface_tension(self):
        "test the surface tension unit"
        self.assertEqual(units.SurfaceTension.base_units(), "[force]/[length]")
        self.assertTrue(pint.Quantity(1, 'N/m').check(units.SurfaceTension.base_units()))


    def test_bulk_modulus(self):
        "test the bulk modulus unit"
        self.assertEqual(units.BulkModulus.base_units(), "[force]/[length]^2")
        self.assertTrue(pint.Quantity(1, 'Pa').check(units.BulkModulus.base_units()))


    def test_length(self):
        "test the length unit"
        self.assertEqual(units.Length.base_units(), "[length]")
        self.assertTrue(pint.Quantity(1, 'm').check(units.Length.base_units()))

    def test_angle(self):
        "test the angle unit"
        self.assertEqual(units.Angle.base_units(), "radians")
        self.assertTrue(pint.Quantity(1, 'rad').check(units.Angle.base_units()))

    def test_area(self):
        "test the area unit"
        self.assertEqual(units.Area.base_units(), "[length]^2")
        self.assertTrue(pint.Quantity(1, 'm^2').check(units.Area.base_units()))

    def test_speed(self):
        "test the speed unit"
        self.assertEqual(units.Speed.base_units(), "[length]/[time]")
        self.assertTrue(pint.Quantity(1, 'm/s').check(units.Speed.base_units()))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelUnits)
    unittest.TextTestRunner(verbosity=2).run(suite)
