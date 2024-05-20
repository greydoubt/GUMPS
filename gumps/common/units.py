# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#this exists to make sure that only one unit registry exists
#Do not import pint and use registry from anywhere else

from pint import UnitRegistry

unit_registry = UnitRegistry()
