import httpx
import pytest

from smartdesk.config import get_settings
from smartdesk.tickets.base import TicketNotFoundError, TicketProviderError
from smartdesk.tickets.jira_provider import JiraProvider
from smartdesk.tickets.models import TicketCreateRequest

BASE = "https://jira.example.test"


def _provider(handler, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("JIRA_BASE_URL", BASE)
    monkeypatch.setenv("JIRA_EMAIL", "admin@example.test")
    monkeypatch.setenv("JIRA_API_TOKEN", "test-token")
    get_settings.cache_clear()
    client = httpx.Client(
        base_url=BASE,
        transport=httpx.MockTransport(handler),
    )
    return JiraProvider(client=client)


def test_create_succeeds_even_if_the_read_back_fails(monkeypatch):
    """
    If Jira created the issue but the follow-up GET fails, we still return the ticket.

    WHY IT MATTERS: reporting failure would make the employee retry and create a
    duplicate ticket for the same problem.
    """
    def handler(request):
        if request.method == "POST":
            return httpx.Response(201, json={"key": "HELP-8"})
        return httpx.Response(503)

    t = _provider(handler, monkeypatch).create_ticket(
        TicketCreateRequest(requester_email="jane@corp.com", title="VPN fails", description="x")
    )
    assert t.id == "HELP-8" and t.url == f"{BASE}/browse/HELP-8"


def test_create_endpoint_404_is_reported_as_provider_error(monkeypatch):
    provider = _provider(
        lambda request: httpx.Response(404, json={"errorMessages": ["Not found"]}),
        monkeypatch,
    )

    with pytest.raises(TicketProviderError, match="endpoint or resource was not found"):
        provider.create_ticket(
            TicketCreateRequest(
                requester_email="jane@corp.com", title="VPN fails", description="x"
            )
        )


def test_get_issue_404_remains_ticket_not_found(monkeypatch):
    provider = _provider(lambda request: httpx.Response(404), monkeypatch)

    with pytest.raises(TicketNotFoundError):
        provider.get_ticket("HELP-404", "jane@corp.com")


def test_bad_request_includes_jira_validation_detail(monkeypatch):
    provider = _provider(
        lambda request: httpx.Response(
            400,
            json={
                "errorMessages": ["Project key is invalid"],
                "errors": {"issuetype": "Issue type is not available"},
            },
        ),
        monkeypatch,
    )

    with pytest.raises(
        TicketProviderError,
        match="Project key is invalid; issuetype: Issue type is not available",
    ):
        provider.create_ticket(
            TicketCreateRequest(
                requester_email="jane@corp.com", title="VPN fails", description="x"
            )
        )