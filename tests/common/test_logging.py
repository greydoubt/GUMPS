# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.common.logging import LoggingContext
import logging

def test_LoggingContext(caplog):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    with LoggingContext(logger, level=logging.WARNING):
        logger.info("info")
        logger.warning("warning")

    assert logger.level == logging.INFO

    # Check the captured logs
    assert "info" not in caplog.text
    assert "warning" in caplog.text

def test_LoggingContext_handler(caplog):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    with LoggingContext(logger, handler=handler):
        logger.info("info")
        logger.warning("warning")

    assert logger.level == logging.INFO

    # Check the captured logs
    assert "info" in caplog.text
    assert "warning" in caplog.text