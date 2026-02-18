"""Retry utility for model queries, thin wrapper around tenacity."""

import logging
import os

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def retry(*, logger: logging.Logger, abort_exceptions: list[type[Exception]]) -> Retrying:
    """Return a tenacity Retrying object with exponential backoff.

    Retries all exceptions except those in abort_exceptions.
    """
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MINIPROPHET_MODEL_RETRY_ATTEMPTS", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )
