# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"This is a stupid process that just sleeps for n seconds, used for testing"

import sys
import time

def test(n):
    "simple function that just sleeps"
    time.sleep(n)

if __name__ == "__main__":
    n = int(sys.argv[1])
    test(n)