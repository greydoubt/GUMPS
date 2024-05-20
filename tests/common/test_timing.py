# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import time

from gumps.common import timing

class TestTimer(unittest.TestCase):
    "test the new kernel interface"

    def test_timer(self):
        timer = timing.Timer()
        target = 0.5
        with timer:
            time.sleep(target)
        elapsed = timer.elapsed()
        self.assertAlmostEqual(elapsed, target, places=1)

    def  test_log_timer(self):
        timer = timing.Timer()
        target = 0.1
        with self.assertLogs('gumps.common.timing', level='DEBUG') as cm:
            with timer:
                time.sleep(target)
        self.assertEqual(cm.output, [f'DEBUG:gumps.common.timing:elapsed time is {timer.elapsed():.2f}'])

    def test_timeout_ok(self):
        time_out = timing.Timeout(1.0)
        target = 0.5
        with time_out:
            time.sleep(target)
        self.assertFalse(time_out.timed_out)

    def test_timeout_over(self):
        time_out = timing.Timeout(1.0)
        target = 1.5
        with time_out:
            time.sleep(target)
        self.assertTrue(time_out.timed_out)

    def test_log_timeout_true(self):
        time_out = timing.Timeout(0.1)
        target = 0.2
        with self.assertLogs('gumps.common.timing', level='DEBUG') as cm:
            with time_out:
                time.sleep(target)
        self.assertEqual(cm.output, ['DEBUG:gumps.common.timing:leaving timeout expired: True'])

    def test_log_timeout_false(self):
        time_out = timing.Timeout(0.1)
        target = 0.01
        with self.assertLogs('gumps.common.timing', level='DEBUG') as cm:
            with time_out:
                time.sleep(target)
        self.assertEqual(cm.output, ['DEBUG:gumps.common.timing:leaving timeout expired: False'])

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTimer)
    unittest.TextTestRunner(verbosity=2).run(suite)
