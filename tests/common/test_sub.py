# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import sys
from pathlib import Path

from gumps.common import sub
from gumps.common import timing
import time

class TestSub(unittest.TestCase):
    "test the new kernel interface"

    def test_run_sub(self):
        "test that the subprocess runs"
        line = [
            sys.executable,
            (Path(__file__).parent / "data" / "dummy.py").as_posix(),
            "1"
        ]
        timer = timing.Timer()
        with timer:
            sub.run_sub("dummy", line, "dummy.py")
            sub.wait_sub("dummy", "dummy.py")
        elapsed = timer.elapsed()
        self.assertTrue(elapsed < 5)

    def test_process_sub(self):
        "test that just checking on the subprocess is nearly instant"
        line = [
            sys.executable,
            (Path(__file__).parent / "data" / "dummy.py").as_posix(),
            "1"
        ]
        sub.run_sub("dummy", line, "dummy.py")

        timer = timing.Timer()
        with timer:
            sub.process_sub("dummy", "dummy.py")
        elapsed = timer.elapsed()
        self.assertTrue(elapsed < 1)
        sub.wait_sub("dummy", "dummy.py")

    def test_log_subprocess(self):
        "test the log function"
        name = "test"
        test = "\n".join(['test1', 'test2', 'test3'])

        correct_output = ['INFO:gumps.common.sub:test stdout: test1',
            'INFO:gumps.common.sub:test stdout: test2',
            'INFO:gumps.common.sub:test stdout: test3',
            'ERROR:gumps.common.sub:test stderr: test1',
            'ERROR:gumps.common.sub:test stderr: test2',
            'ERROR:gumps.common.sub:test stderr: test3',]



        with self.assertLogs('gumps.common.sub', level='INFO') as cm:
            sub.log_subprocess(name, test, test)

        self.assertEqual(cm.output, correct_output)

    def test_log_run_process(self):
        line = [
            sys.executable,
            (Path(__file__).parent.parent / "data" / "dummy.py").as_posix(),
            "0"
        ]
        with self.assertLogs('gumps.common.sub', level='INFO') as cm:
            sub.run_sub("dummy", line, "dummy.py")
        self.assertEqual(cm.output, ['INFO:gumps.common.sub:creating subprocess dummy for dummy.py'])

        #we need to sleep long enough that the subprocess has returned to get the log entry
        #from process
        time.sleep(4)

        with self.assertLogs('gumps.common.sub', level='INFO') as cm:
            sub.process_sub("dummy", "dummy.py")
        self.assertEqual(cm.output, ['INFO:gumps.common.sub:finished subprocess dummy for dummy.py'])

        sub.wait_sub("dummy", "dummy.py")

    def test_log_wait(self):
        line = [
            sys.executable,
            (Path(__file__).parent.parent / "data" / "dummy.py").as_posix(),
            "4"
        ]
        sub.run_sub("dummy", line, "dummy.py")

        with self.assertLogs('gumps.common.sub', level='INFO') as cm:
            sub.wait_sub("dummy", "dummy.py")
        self.assertEqual(cm.output, ['INFO:gumps.common.sub:waiting for subprocess dummy for dummy.py',
            'INFO:gumps.common.sub:finished subprocess dummy for dummy.py'])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSub)
    unittest.TextTestRunner(verbosity=2).run(suite)
