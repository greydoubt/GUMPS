# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

class GUMPSException(Exception):
    "Base class for all GUMPS exceptions"

class IllDefinedException(GUMPSException):
    "Raise Error if problem is ill-defined"

class NoLPointFoundError(GUMPSException):
    "Raise error if L-point distance is negative"
