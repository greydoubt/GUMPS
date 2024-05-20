# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import random
import os

#this is to make the tests predictable, DO NOT REMOVE
random.seed(0)

from gumps.common import parallel

def make_pi(count):
    "calculate pi"
    random.seed(0)
    count_inbound = 0
    for x in range(count):
        the_x = random.random()
        the_y = random.random()
        if((the_x**2 + the_y**2) <= 1):
            count_inbound += 1
    return count_inbound

def make_pi_exception(count):
    "calculate pi"
    raise RuntimeError("This is an exception")


def handle_exception(function, x, e):
    "handle the exception"
    return None

def handle_exception_raise(function, x, e):
    "handle the exception"
    return e

class TestParallel(unittest.TestCase):
    "test the multiprocessing parallel interface"

    def test_sequential(self):
        pool_size = 1
        pieces = 4
        count = 100
        with parallel.Parallel(poolsize=pool_size) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)

    def test_parallel(self):
        pool_size = 4
        pieces = 4
        count = 100
        with parallel.Parallel(poolsize=pool_size) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)


    def test_parallel_exception(self):
        pool_size = 4
        pieces = 4
        count = 8
        with parallel.Parallel(poolsize=pool_size) as map_function:
            with self.assertRaises(Exception):
                list(map_function(make_pi_exception, [count for x in range(pieces)]))

    def test_pool_size(self):
        "test the pool size"
        pool_size = None
        target_pool_size = os.cpu_count() or 1
        pool = parallel.Parallel(poolsize=pool_size)
        with pool:
            self.assertEqual(pool.poolsize, target_pool_size)

    def test_parallel_unordered(self):
        pool_size = 4
        pieces = 4
        count = 100
        with parallel.Parallel(poolsize=pool_size, ordered=False) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)

    def test_parallel_max_tasks(self):
        pool_size = 4
        pieces = 8
        count = 100
        with parallel.Parallel(poolsize=pool_size, maxtasksperchild=1) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)

    def test_log_sequential_start(self):
        with self.assertLogs('gumps.common.parallel', level='INFO') as cm:
            with parallel.Parallel(poolsize=1):
                pass
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using default map and no parallelization'])

    def test_log_parallel_start(self):
        with self.assertLogs('gumps.common.parallel', level='INFO') as cm:
            with parallel.Parallel(poolsize=2):
                pass
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel pool with 2 workers'])

    def test_get_iterator_map(self):
        "test the get iterator function"
        pool = parallel.Parallel(poolsize=1)
        with pool:
            self.assertEqual(pool.get_iterator(), map)

    def test_get_iterator_imap(self):
        "test the get iterator function"
        pool = parallel.Parallel(poolsize=2)
        with pool:
            self.assertEqual(pool.get_iterator(), pool.pool.imap)

    def test_get_iterator_imap_unordered(self):
        "test the get iterator function"
        pool = parallel.Parallel(poolsize=2, ordered=False)
        with pool:
            self.assertEqual(pool.get_iterator(), pool.pool.imap_unordered)

    def test_parallel_runner_imap(self):
        pool_size = 4
        pieces = 4
        count = 100
        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with parallel.Parallel(poolsize=pool_size) as map_function:
                count_in = map_function(make_pi, [count for x in range(pieces)])
                pi = 4*sum(count_in)/(count * pieces)
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel pool with 4 workers',
            'DEBUG:gumps.common.parallel:using imap',
            'DEBUG:gumps.common.parallel:parallel pool closing down'])

    def test_parallel_runner_imap_unordered(self):
        pool_size = 4
        pieces = 4
        count = 100
        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with parallel.Parallel(poolsize=pool_size, ordered=False) as map_function:
                count_in = map_function(make_pi, [count for x in range(pieces)])
                pi = 4*sum(count_in)/(count * pieces)
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel pool with 4 workers',
            'DEBUG:gumps.common.parallel:using imap_unordered',
            'DEBUG:gumps.common.parallel:parallel pool closing down'])

    def test_parallel_runner_map(self):
        pool_size = 1
        pieces = 4
        count = 100
        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with parallel.Parallel(poolsize=pool_size) as map_function:
                count_in = map_function(make_pi, [count for x in range(pieces)])
                pi = 4*sum(count_in)/(count * pieces)
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using default map and no parallelization',
            'DEBUG:gumps.common.parallel:using map function'])


class TestLambdaParallel(unittest.TestCase):
    "test the lambda parallel interface"

    def test_sequential(self):
        pool_size = 1
        pieces = 4
        count = 100
        with parallel.LambdaParallel(poolsize=pool_size) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)

    def test_parallel(self):
        pool_size = 4
        pieces = 4
        count = 100
        with parallel.LambdaParallel(poolsize=pool_size) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)

    def test_pool_size(self):
        "test the pool size"
        pool_size = None
        target_pool_size = os.cpu_count() or 1
        pool = parallel.LambdaParallel(poolsize=pool_size)
        with pool:
            self.assertEqual(pool.poolsize, target_pool_size)

    def test_log_sequential_start(self):
        with self.assertLogs('gumps.common.parallel', level='INFO') as cm:
            with parallel.LambdaParallel(poolsize=1):
                pass
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using default map and no parallelization'])

    def test_log_parallel_start(self):
        with self.assertLogs('gumps.common.parallel', level='INFO') as cm:
            with parallel.LambdaParallel(poolsize=2):
                pass
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel pool with 2 workers'])

    def test_parallel_runner_parallel_map(self):
        pool_size = 4
        pieces = 4
        count = 100
        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with parallel.LambdaParallel(poolsize=pool_size) as map_function:
                count_in = map_function(make_pi, [count for x in range(pieces)])
                pi = 4*sum(count_in)/(count * pieces)
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel pool with 4 workers',
            'DEBUG:gumps.common.parallel:using lambda map',
            'DEBUG:gumps.common.parallel:parallel pool closing down'])

    def test_parallel_runner_default_map(self):
        "test the parallel running map function"
        pool_size = 1
        pieces = 4
        count = 100
        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with parallel.LambdaParallel(poolsize=pool_size) as map_function:
                count_in = map_function(make_pi, [count for x in range(pieces)])
                pi = 4*sum(count_in)/(count * pieces)
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using default map and no parallelization',
            'DEBUG:gumps.common.parallel:using map function'])


class TestMultiprocessParallel(unittest.TestCase):
    "test the multiprocessing parallel interface"

    def test_sequential(self):
        pool_size = 1
        pieces = 4
        count = 100
        with parallel.MultiprocessParallel(poolsize=pool_size) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)

    def test_parallel(self):
        pool_size = 4
        pieces = 4
        count = 100
        with parallel.MultiprocessParallel(poolsize=pool_size) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)


    def test_parallel_exception(self):
        pool_size = 4
        pieces = 4
        count = 8
        with parallel.MultiprocessParallel(poolsize=pool_size) as map_function:
            with self.assertRaises(Exception):
                list(map_function(make_pi_exception, [count for x in range(pieces)]))

    def test_pool_size(self):
        "test the pool size"
        pool_size = None
        target_pool_size = os.cpu_count() or 1
        pool = parallel.MultiprocessParallel(poolsize=pool_size)
        with pool:
            self.assertEqual(pool.poolsize, target_pool_size)

    def test_parallel_unordered(self):
        pool_size = 4
        pieces = 4
        count = 100
        with parallel.MultiprocessParallel(poolsize=pool_size, ordered=False) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)

    def test_parallel_max_tasks(self):
        pool_size = 4
        pieces = 8
        count = 100
        with parallel.MultiprocessParallel(poolsize=pool_size, maxtasksperchild=1) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)

    def test_log_sequential_start(self):
        with self.assertLogs('gumps.common.parallel', level='INFO') as cm:
            with parallel.MultiprocessParallel(poolsize=1):
                pass
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using default map and no parallelization'])

    def test_log_parallel_start(self):
        with self.assertLogs('gumps.common.parallel', level='INFO') as cm:
            with parallel.MultiprocessParallel(poolsize=2):
                pass
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel pool with 2 workers'])

    def test_get_iterator_map(self):
        "test the get iterator function"
        pool = parallel.MultiprocessParallel(poolsize=1)
        with pool:
            self.assertEqual(pool.get_iterator(), map)

    def test_get_iterator_imap(self):
        "test the get iterator function"
        pool = parallel.MultiprocessParallel(poolsize=2)
        with pool:
            self.assertEqual(pool.get_iterator(), pool.pool.imap)

    def test_get_iterator_imap_unordered(self):
        "test the get iterator function"
        pool = parallel.MultiprocessParallel(poolsize=2, ordered=False)
        with pool:
            self.assertEqual(pool.get_iterator(), pool.pool.imap_unordered)

    def test_parallel_runner_imap(self):
        pool_size = 4
        pieces = 4
        count = 100
        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with parallel.MultiprocessParallel(poolsize=pool_size) as map_function:
                count_in = map_function(make_pi, [count for x in range(pieces)])
                pi = 4*sum(count_in)/(count * pieces)
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel pool with 4 workers',
            'DEBUG:gumps.common.parallel:using imap',
            'DEBUG:gumps.common.parallel:parallel pool closing down'])

    def test_parallel_runner_imap_unordered(self):
        pool_size = 4
        pieces = 4
        count = 100
        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with parallel.MultiprocessParallel(poolsize=pool_size, ordered=False) as map_function:
                count_in = map_function(make_pi, [count for x in range(pieces)])
                pi = 4*sum(count_in)/(count * pieces)
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel pool with 4 workers',
            'DEBUG:gumps.common.parallel:using imap_unordered',
            'DEBUG:gumps.common.parallel:parallel pool closing down'])

    def test_parallel_runner_map(self):
        pool_size = 1
        pieces = 4
        count = 100
        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with parallel.MultiprocessParallel(poolsize=pool_size) as map_function:
                count_in = map_function(make_pi, [count for x in range(pieces)])
                pi = 4*sum(count_in)/(count * pieces)
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using default map and no parallelization',
            'DEBUG:gumps.common.parallel:using map function'])


class TestThreadParallel(unittest.TestCase):
    "test the thread parallel interface"

    def test_sequential(self):
        pool_size = 1
        pieces = 4
        count = 100
        with parallel.ThreadParallel(poolsize=pool_size) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)

    def test_parallel(self):
        pool_size = 4
        pieces = 4
        count = 100
        with parallel.ThreadParallel(poolsize=pool_size) as map_function:
            count_in = map_function(make_pi, [count for x in range(pieces)])
            pi = 4*sum(count_in)/(count * pieces)
            self.assertAlmostEqual(pi, 3, delta=0.25)

    def test_pool_size(self):
        "test the pool size"
        pool_size = None
        target_pool_size = os.cpu_count() or 1
        pool = parallel.ThreadParallel(poolsize=pool_size)
        with pool:
            self.assertEqual(pool.poolsize, target_pool_size)

    def test_log_sequential_start(self):
        with self.assertLogs('gumps.common.parallel', level='INFO') as cm:
            with parallel.ThreadParallel(poolsize=1):
                pass
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using default map and no parallelization'])

    def test_log_parallel_start(self):
        with self.assertLogs('gumps.common.parallel', level='INFO') as cm:
            with parallel.ThreadParallel(poolsize=2):
                pass
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel thread pool with 2 workers'])

    def test_parallel_runner_parallel_map(self):
        pool_size = 4
        pieces = 4
        count = 100
        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with parallel.ThreadParallel(poolsize=pool_size) as map_function:
                count_in = map_function(make_pi, [count for x in range(pieces)])
                pi = 4*sum(count_in)/(count * pieces)
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel thread pool with 4 workers',
            'DEBUG:gumps.common.parallel:using imap',
            'DEBUG:gumps.common.parallel:parallel pool closing down'])

    def test_parallel_runner_default_map(self):
        "test the parallel running map function"
        pool_size = 1
        pieces = 4
        count = 100
        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with parallel.ThreadParallel(poolsize=pool_size) as map_function:
                count_in = map_function(make_pi, [count for x in range(pieces)])
                pi = 4*sum(count_in)/(count * pieces)
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using default map and no parallelization',
            'DEBUG:gumps.common.parallel:using map function'])

    def test_get_iterator_imap_unordered(self):
        "test the get iterator function"
        pool = parallel.ThreadParallel(poolsize=2, ordered=False)
        with pool:
            self.assertEqual(pool.get_iterator(), pool.pool.imap_unordered)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParallel)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestLambdaParallel)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiprocessParallel)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestThreadParallel)
    unittest.TextTestRunner(verbosity=2).run(suite)
