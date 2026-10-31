"""
Live check against a SANDBOX Jira project (the unit tests use a fake transport,
so this is the only thing that proves your credentials, project key and issue
type really work).

Usage:  TICKET_PROVIDER=jira python scripts/jira_smoke_test.py you@corp.com
"""

import sys

from smartdesk.config import get_settings
from smartdesk.tickets.factory import get_ticket_provider
from smartdesk.tickets.tools import create_support_ticket, lookup_tickets


def main() -> None:
    """Create one ticket, then read it back by ID and by email."""
    if get_settings().ticket_provider != "jira":
        sys.exit("Set TICKET_PROVIDER=jira first, or this would test the mock.")
    email = sys.argv[1]
    provider = get_ticket_provider()

    created = create_support_ticket(provider, email, "SmartDesk smoke test", "Safe to delete.")
    if not created.ok:
        sys.exit(f"Create failed: {created.error}: {created.detail}")
    print("Created:", created.ticket.id, created.ticket.url)

    by_id = lookup_tickets(provider, email, ticket_id=created.ticket.id)
    by_email = lookup_tickets(provider, email)
    print("By ID:   ", by_id.outcome, [t.raw_status for t in by_id.tickets])
    print("By email:", by_email.outcome, [t.id for t in by_email.tickets])


if __name__ == "__main__":
    main()