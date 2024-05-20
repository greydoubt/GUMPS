# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import numpy as np
import pandas as pd

from gumps.kernels.ackley_complete_kernel import AckleyCompleteKernel
from gumps.studies.ackley_batch_study import AckleyBatchStudy
from gumps.studies.study import SimpleSimulationStudy


class TestAckleyBatchStudy(unittest.TestCase):
    "test the new study interface"

    def test_batch_ackley(self):
        "test parallel running"
        batch = AckleyBatchStudy(model_variables={})

        input_data = pd.DataFrame({'x': [np.array([1,2,3,4]),np.array([1,2,3,4]),np.array([1,2,3,4])],
            'a' : [1,2,3],
            'b' : [1,2,3],
            'c' : [1,2,3],
        })

        def get_total(frame:pd.DataFrame):
            return pd.DataFrame(frame['total'])

        with batch:
            totals_batch = batch.run(input_data, get_total)

        kernel = AckleyCompleteKernel()
        totals = []
        for _,problem in input_data.iterrows():
            problem = problem.to_dict()
            study = SimpleSimulationStudy(problem=problem, kernel=kernel)
            study.run()
            result = study.state_frame()
            totals.append(result['total'][0])

        correct = pd.DataFrame({'total': totals})

        pd.testing.assert_frame_equal(totals_batch, correct)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAckleyBatchStudy)
    unittest.TextTestRunner(verbosity=2).run(suite)