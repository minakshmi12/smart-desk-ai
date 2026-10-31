"""
Logging setup, called once at application startup.

WHY THIS FILE EXISTS
--------------------
Every module uses logging.getLogger(__name__). Without one place that configures
the output format and level, those messages are silently dropped by the server.

PRIVACY NOTE: the code logs events and error types, never message text. Chat
content can contain personal details, and logs are usually retained and shared
more widely than the tickets are.
"""

import logging


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger.

    Args:
        level: a standard level name ("DEBUG", "INFO", "WARNING"...). An unknown
            name falls back to INFO instead of crashing startup.

    WHY force=True: uvicorn may already have installed handlers. Forcing makes
    our format apply consistently and keeps repeated calls (tests, reloads) from
    stacking duplicate handlers that print every line twice.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
