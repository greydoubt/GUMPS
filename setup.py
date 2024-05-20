# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import setup, find_packages

setup(name = 'gumps',
	version = '3.1.3',
	packages = find_packages(),
	ext_modules = [],
	description = 'Generic universal modeling platform software',
	author = 'William Heymann, Joao Alberto de Faria',
	maintainer_email = '',
	install_requires = [
        'lambda-multiprocessing>=0.3'
	],
	python_requires='>=3.10',
    license="BSD-3-Clause",
	)
