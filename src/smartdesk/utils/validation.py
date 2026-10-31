"""
Hard validation rules enforced in code (never delegated to the LLM).

WHY THIS FILE EXISTS
--------------------
The LLM extracts an email from free text (EmailExtraction), but whether it is
*valid* is a yes/no rule that code can decide deterministically. Keeping it here
gives the ticket node, the models, and the tests one shared definition.
"""

import re

# Deliberately simple: one "@", no spaces, a dot in the domain. Full RFC 5322
# validation is overkill and rejects real addresses. The real proof an email is
# right is that a human can later be reached at it.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)+$")


def normalize_email(email: str) -> str:
    """
    Trim whitespace and lowercase an email.

    WHY: "Jane@Corp.com " and "jane@corp.com" are the same person. Normalizing
    before storing or searching means "find my tickets" matches however the
    employee typed it.
    """
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    """Return True if `email` looks like a well-formed address (after normalizing)."""
    return bool(_EMAIL_RE.match(normalize_email(email)))