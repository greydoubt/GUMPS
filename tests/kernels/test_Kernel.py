# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import attrs
from gumps.kernels import AbstractKernel

@attrs.define
class TestKernelState:
    a:int = attrs.field(default=1)
    b:list[float] = attrs.field(default=[1.0,2.0,3.0])
    c:None = attrs.field(default=None)


class TestKernel(unittest.TestCase):
    "test the new kernel interface"
    def test_abstract_interface(self):
        with self.assertRaises(TypeError):
            AbstractKernel()

    def test_kernel_function(self):
        "test a missing user_defined_function method"
        class Kernel(AbstractKernel):
            def get_state_class(self) -> TestKernelState:
                return TestKernelState
        with self.assertRaises(TypeError):
            Kernel({})

    def test_kernel_states(self):
        "test a missing get_states method"
        class Kernel(AbstractKernel):
            def user_defined_function(self, variables: TestKernelState):
                pass
        with self.assertRaises(TypeError):
            Kernel({})

    def test_kernel_success(self):
        "test a complete class can be created"
        class Kernel(AbstractKernel):
            def user_defined_function(self, variables: TestKernelState):
                pass
            def get_state_class(self) -> TestKernelState:
                return TestKernelState
        Kernel({})

    def test_log_f(self):
        "test a complete class can be created"
        class Kernel(AbstractKernel):
            def user_defined_function(self, variables: TestKernelState):
                pass
            def get_state_class(self) -> TestKernelState:
                return TestKernelState
        kern = Kernel({})

        with self.assertLogs('gumps.kernels.kernel', level='DEBUG') as cm:
            kern.f({})
        self.assertEqual(cm.output, ['DEBUG:gumps.kernels.kernel:creating a new state class and calling user_defined_function'])

    def test_create_class(self):
        "test that an attr class is created successfully"
        class Kernel(AbstractKernel):
            def initialize(self):
                self.model = {'a':1, 'b':[1.0,2.0,3.0], 'c':None}
            def user_defined_function(self, variables: TestKernelState):
                pass
            def get_state_class(self) -> TestKernelState:
                return TestKernelState

        @attrs.define
        class Test:
            a:int = attrs.field(default=1)
            b:list[float] = attrs.field(default=[1.0,2.0,3.0])
            c:None = attrs.field(default=None)

        allowed_states = set(['a', 'b', 'c'])
        model_variables = {'a':1, 'b':2, 'c':3}

        with self.assertLogs('gumps.kernels.kernel', level='DEBUG') as cm:
            testKernel = Kernel(model_variables=model_variables)
        self.assertEqual(cm.output, [f'DEBUG:gumps.kernels.kernel:states found {set(testKernel.allowed_state)}'])

        with self.subTest("allowed_state"):
            self.assertEqual(testKernel.allowed_state, allowed_states)
        with self.subTest("model_variables"):
            self.assertEqual(testKernel.model_variables, model_variables)
        with self.subTest("state_class"):
            testAttr = Test()
            testState = testKernel.get_state_object({})
            self.assertEqual(testAttr.a, testState.a)
            self.assertEqual(testAttr.b, testState.b)
            self.assertEqual(testAttr.c, testState.c)

    def test_new_kernel(self):
        "test that new kernel can create a new kernel with updated values"

        class Kernel(AbstractKernel):
            def initialize(self):
                self.model = {'a':1, 'b':[1.0,2.0,3.0], 'c':None, 'total': 0}
            def user_defined_function(self, variables: TestKernelState):
                pass
            def get_state_class(self) -> TestKernelState:
                return TestKernelState

        model_variables = {'a':1, 'b':2, 'c':3}
        testKernel = Kernel(model_variables=model_variables)


        newKernel = testKernel.new_kernel({'a':5, 'c':4})
        self.assertDictEqual(newKernel.model_variables, {'a': 5, 'b': 2, 'c': 4})

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKernel)
    unittest.TextTestRunner(verbosity=2).run(suite)
