# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import tempfile
import os
import copy

from gumps.common import hdf5
from addict import Dict
import numpy as np

class TestHDF5(unittest.TestCase):
    "test the new kernel interface"

    def setUp(self):
        "create some nested dictionaries for use with testing"
        self.good = Dict()
        self.good.foo.bar.string = "test"
        self.good.foo.bar.integer = 1
        self.good.foo.bar.float = 1.0
        self.good.foo.bar.tuple_int = (1,2,3,4,5)
        self.good.foo.bar.tuple_float = (1.0,2.0,3.0,4.0,5.0)
        self.good.foo.list_int = (1,2,3,4,5)
        self.good.foo.list_float = (1.0,2.0,3.0,4.0,5.0)
        self.good.baz.array_int = np.array([1,2,3,4,5])
        self.good.baz.array_float = np.array([1.0,2.0,3.0,4.0,5.0])


    def compare(self, name, data, correct):
        "compare data to good"
        with self.subTest(f"{name} string"):
            self.assertEqual(correct.foo.bar.string, data.root.foo.bar.string)

        with self.subTest(f"{name} integer"):
            self.assertEqual(correct.foo.bar.integer, data.root.foo.bar.integer)

        with self.subTest(f"{name} float"):
            self.assertEqual(correct.foo.bar.float, data.root.foo.bar.float)

        with self.subTest(f"{name} tuple_int"):
            np.testing.assert_array_equal(correct.foo.bar.tuple_int, data.root.foo.bar.tuple_int)

        with self.subTest(f"{name} tuple_float"):
            np.testing.assert_array_equal(correct.foo.bar.tuple_float, data.root.foo.bar.tuple_float)

        with self.subTest(f"{name} list_int"):
            try:
                np.testing.assert_array_equal(correct.foo.list_int, data.root.foo.list_int)
            except ValueError:
                self.assertEqual(correct.foo.list_int, data.root.foo.list_int)

        with self.subTest(f"{name} list_float"):
            try:
                np.testing.assert_array_equal(correct.foo.list_float, data.root.foo.list_float)
            except ValueError:
                self.assertEqual(correct.foo.list_float, data.root.foo.list_float)

        with self.subTest(f"{name} array_int"):
            np.testing.assert_array_equal(correct.baz.array_int, data.root.baz.array_int)

        with self.subTest(f"{name} array_float"):
            np.testing.assert_array_equal(correct.baz.array_float, data.root.baz.array_float)


    def test_except_save_bad_name(self):
        #create the file
        data = hdf5.H5('')
        data.root = self.good
        with self.assertRaises((ValueError, OSError)):
            data.save(lock=True)

    def test_except_load_notfound(self):
        #create the file
        data = hdf5.H5('foo.h5')
        with self.assertRaises(FileNotFoundError):
            data.load(lock=True)

    def test_except_load_bad_name(self):
        #create the file
        data = hdf5.H5('')
        with self.assertRaises((ValueError, OSError)):
            data.load(lock=True)

    def test_json_except_save_bad_name(self):
        #create the file
        data = hdf5.H5('')
        data.root = self.good
        with self.assertRaises((PermissionError, IsADirectoryError)):
            data.save_json()

    def test_json_except_load_notfound(self):
        #create the file
        data = hdf5.H5('foo.h5')
        with self.assertRaises(FileNotFoundError):
            data.load_json()

    def test_json_except_load_bad_name(self):
        #create the file
        data = hdf5.H5('')
        with self.assertRaises((PermissionError, IsADirectoryError)):
            data.load_json()

    def test_save_load_lock_true(self):
        handle, path = tempfile.mkstemp(suffix='.h5')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        data.save(lock=True)

        #load the file
        data = hdf5.H5(path)
        data.load(lock=True)

        os.remove(path)

        #everything comes back as numpy types except for strings
        self.compare('load_lock_true', data, self.good)

    def test_log_hdf5_save_load(self):
        handle, path = tempfile.mkstemp(suffix='.h5')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        with self.assertLogs('gumps.common.hdf5', level='DEBUG') as cm:
            data.save(lock=True)
        self.assertEqual(cm.output, [f'DEBUG:gumps.common.hdf5:saving hdf5 to {path}'])

        #load the file
        data = hdf5.H5(path)
        with self.assertLogs('gumps.common.hdf5', level='DEBUG') as cm:
            data.load(lock=True)
        self.assertEqual(cm.output, [f'DEBUG:gumps.common.hdf5:loading hdf5 from {path}'])

        os.remove(path)

    def test_log_json_save_load(self):
        handle, path = tempfile.mkstemp(suffix='.json')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        with self.assertLogs('gumps.common.hdf5', level='DEBUG') as cm:
            data.save_json()
        self.assertEqual(cm.output, [f'DEBUG:gumps.common.hdf5:saving json to {path}'])

        #load the file
        data = hdf5.H5(path)
        with self.assertLogs('gumps.common.hdf5', level='DEBUG') as cm:
            data.load_json()
        self.assertEqual(cm.output, [f'DEBUG:gumps.common.hdf5:loading json from {path}'])

        os.remove(path)


    def test_save_load_lock_false(self):
        handle, path = tempfile.mkstemp(suffix='.h5')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        data.save(lock=False)

        #load the file
        data = hdf5.H5(path)
        data.load(lock=False)

        os.remove(path)

        #everything comes back as numpy types except for strings
        self.compare('load_lock_true', data, self.good)

    def test_load_update(self):
        handle, path = tempfile.mkstemp(suffix='.h5')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        data.save()

        #load the file
        data = hdf5.H5(path)
        data.root.foo.here = 1
        data.root.foo.bar.here = 1
        data.load(lock=True, update=True)

        os.remove(path)

        #everything comes back as numpy types except for strings
        self.compare('load_lock_true', data, self.good)

        with self.subTest("foo here"):
            self.assertEqual(data.root.foo.here, 1)

        with self.subTest("foo bar here"):
            self.assertEqual(data.root.foo.bar.here, 1)

    def test_load_paths(self):
        handle, path = tempfile.mkstemp(suffix='.h5')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        data.save()

        #load the file
        data = hdf5.H5(path)
        data.load(lock=True, paths=['/foo/bar', '/baz'])

        os.remove(path)

        good = copy.deepcopy(self.good)
        del good.foo.list_int
        del good.foo.list_float
        self.compare('load_lock_true', data, good)

    def test_save_load_json(self):
        handle, path = tempfile.mkstemp(suffix='.json')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        data.save_json()

        #load the file
        data = hdf5.H5(path)
        data.load_json()

        os.remove(path)

        #everything comes back as numpy types except for strings
        self.compare('load_lock_true', data, self.good)

    def test_load_json_update(self):
        handle, path = tempfile.mkstemp(suffix='.json')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        data.save_json()

        #load the file
        data = hdf5.H5(path)
        data.root.foo.here = 1
        data.root.foo.bar.here = 1
        data.load_json(update=True)

        os.remove(path)

        #everything comes back as numpy types except for strings
        self.compare('load_lock_true', data, self.good)

        with self.subTest("foo here"):
            self.assertEqual(data.root.foo.here, 1)

        with self.subTest("foo bar here"):
            self.assertEqual(data.root.foo.bar.here, 1)

    def test_append(self):
        handle, path = tempfile.mkstemp(suffix='.h5')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        data.save()

        #append
        data = hdf5.H5(path)
        data.root.foo.here = 1
        data.root.foo.bar.here = 1
        data.append()

        #load the file
        data = hdf5.H5(path)
        data.load(lock=True)

        os.remove(path)

        #everything comes back as numpy types except for strings
        self.compare('load_lock_true', data, self.good)

        with self.subTest("foo here"):
            self.assertEqual(data.root.foo.here, 1)

        with self.subTest("foo bar here"):
            self.assertEqual(data.root.foo.bar.here, 1)

    def test_append_lock(self):
        handle, path = tempfile.mkstemp(suffix='.h5')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        data.save()

        #append
        data = hdf5.H5(path)
        data.root.foo.here = 1
        data.root.foo.bar.here = 1
        data.append(lock=True)

        #load the file
        data = hdf5.H5(path)
        data.load(lock=True)

        os.remove(path)

        #everything comes back as numpy types except for strings
        self.compare('load_lock_true', data, self.good)

        with self.subTest("foo here"):
            self.assertEqual(data.root.foo.here, 1)

        with self.subTest("foo bar here"):
            self.assertEqual(data.root.foo.bar.here, 1)

    def test_log_hdf5_append(self):
        handle, path = tempfile.mkstemp(suffix='.h5')
        os.close(handle)

        #create the file
        data = hdf5.H5(path)
        data.root = self.good
        data.save()

        #append
        data = hdf5.H5(path)
        data.root.foo.here = 1
        data.root.foo.bar.here = 1

        os.remove(path)

        with self.assertLogs('gumps.common.hdf5', level='DEBUG') as cm:
            data.append()
        self.assertEqual(cm.output, [f'DEBUG:gumps.common.hdf5:appending hdf5 to {path}'])

    def test_update(self):
        data = hdf5.H5('')
        data.root.foo.bar.string = "test"
        data.root.foo.bar.integer = 1
        data.root.foo.bar.float = 1.0
        data.root.foo.bar.tuple_int = (1,2,3,4,5)
        data.root.foo.bar.tuple_float = (1.0,2.0,3.0,4.0,5.0)

        data2 = hdf5.H5('')
        data2.root.foo.list_int = (1,2,3,4,5)
        data2.root.foo.list_float = (1.0,2.0,3.0,4.0,5.0)
        data2.root.baz.array_int = np.array([1,2,3,4,5])
        data2.root.baz.array_float = np.array([1.0,2.0,3.0,4.0,5.0])

        data.update(data2)

        self.compare('load_lock_true', data, self.good)

    def test_get_set(self):
        data = hdf5.H5('')
        data['/foo/bar/baz'] = 1

        self.assertEqual(data.root.foo.bar.baz, 1)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHDF5)
    unittest.TextTestRunner(verbosity=2).run(suite)
