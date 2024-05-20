# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import pint
import gumps.common.units

class TestUnits(unittest.TestCase):
    "Test the unit converson"

    def test_unit_conversion(self):
        "check if unit conversion is working correctly"
        ureg = gumps.common.units.unit_registry
        distance = 1 * ureg.meter
        distance_mm = distance.to('millimeter')
        self.assertEqual(str(distance_mm), "1000.0 millimeter" )
        self.assertEqual(distance_mm.magnitude, 1e3)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUnits)
    unittest.TextTestRunner(verbosity=2).run(suite)
