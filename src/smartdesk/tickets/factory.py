"""
Chooses the ticket provider from configuration.

WHY THIS FILE EXISTS
--------------------
Nodes should not contain `if provider == "jira"` logic. They call
get_ticket_provider() and get back *something that implements TicketProvider*.
Switching platforms becomes a one-line change in .env (TICKET_PROVIDER=jira).
"""

from functools import lru_cache

from smartdesk.config import get_settings
from smartdesk.tickets.base import TicketProvider


@lru_cache
def get_ticket_provider() -> TicketProvider:
    """
    Build the configured provider once and reuse it.

    WHY CACHED: JiraProvider holds an httpx.Client with a connection pool;
    reusing it avoids a new TLS handshake on every chat message. In tests, call
    get_ticket_provider.cache_clear() after changing env vars.

    WHY IMPORTS ARE INSIDE THE BRANCHES: with TICKET_PROVIDER=mock you shouldn't
    need Jira/Asana/Notion code (or their future dependencies) to load.
    """
    name = get_settings().ticket_provider
    if name == "mock":
        from smartdesk.tickets.mock_provider import MockTicketProvider
        return MockTicketProvider()
    if name == "jira":
        from smartdesk.tickets.jira_provider import JiraProvider
        return JiraProvider()
    raise NotImplementedError(f"Ticket provider '{name}' is not implemented yet")