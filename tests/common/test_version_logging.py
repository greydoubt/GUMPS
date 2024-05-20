# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

from gumps.common import version_logging

class TestVersionLogging(unittest.TestCase):
    "test the version logging"

    def test_git(self):
        cmd = ("--version")
        out = version_logging.git(cmd)
        self.assertIsNotNone(out)

    def test_get_latest_git_commit_short_true(self):
        'test the latest git commit with short_id=True by default'
        out = version_logging.get_latest_git_commit()
        self.assertIsNotNone(out)

    def test_get_latest_git_commit_short_false(self):
        'test the latest git commit with short_id=False'
        out = version_logging.get_latest_git_commit(short_id=False)
        self.assertIsNotNone(out)

    def test_get_current_git_branch(self):
        
        out = version_logging.get_current_git_branch()
        self.assertIsNotNone(out)


    def test_check_uncomitted_changesh(self):
        
        out = version_logging.check_uncomitted_changes()
        self.assertIsNotNone(out)

    def test_version_logging(self):
        
        with self.assertLogs('gumps.common.version_logging', level='INFO') as cm:
            version_logging.version_logging()

        self.assertIsNotNone(cm.output)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVersionLogging)
    unittest.TextTestRunner(verbosity=2).run(suite)