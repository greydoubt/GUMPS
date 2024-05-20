# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import typing

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

processes: dict[str, typing.Optional[subprocess.Popen]] = {}

def run_sub(key, line, file_name):
    "run a subprocess"
    sub = processes.get(key, None)

    if sub is None:
        logger.info(
            "creating subprocess %s for %s", key, file_name
        )

        sub = subprocess.Popen(
            line,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes[key] = sub

    process_sub(key, file_name)


def process_sub(key, file_name):
    "process sub if sub is complete"
    sub = processes.get(key, None)

    if sub is not None:
        finished = sub.poll() is not None
        if finished is not False:
            del processes[key]
            stdout, stderr = sub.communicate()
            logger.info(f"finished subprocess {key} for {file_name}")
            log_subprocess(file_name, stdout.decode("utf-8"), stderr.decode("utf-8"))


def wait_sub(key, file_name):
    "wait for a subprocess"
    sub = processes.get(key, None)
    if sub is not None:
        logger.info(f"waiting for subprocess {key} for {file_name}")
        stdout, stderr = sub.communicate()
        del processes[key]
        logger.info(f"finished subprocess {key} for {file_name}")
        log_subprocess(file_name, stdout.decode("utf-8"), stderr.decode("utf-8"))


def log_subprocess(name, stdout, stderr):
    "log a subprocess"
    for line in stdout.splitlines():
        logger.info(f"{name} stdout: {line}")

    for line in stderr.splitlines():
        logger.error(f"{name} stderr: {line}")
