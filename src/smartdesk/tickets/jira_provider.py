"""
Jira Cloud implementation of TicketProvider (REST API v3).

WHY THIS FILE EXISTS
--------------------
Translates between SmartDesk's neutral models and Jira's API. All Jira-specific
knowledge (auth, ADF format, JQL, status categories) is confined to this file.

HOW REQUESTERS ARE LINKED
-------------------------
Jira can only set a ticket's "reporter" via an internal accountId, not an email,
and lookup requires extra permissions. Instead, each ticket gets:
  1. a deterministic label like `req-jane_at_corp.com`  -> makes "find my tickets"
     a simple, exact JQL query (labels = "..."),
  2. the email written into the description              -> so the human agent can
     see and contact the employee.
"""

import re
from datetime import datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from smartdesk.config import get_settings
from smartdesk.tickets.base import TicketNotFoundError, TicketProvider, TicketProviderError
from smartdesk.tickets.models import Ticket, TicketCreateRequest, TicketStatus
from smartdesk.utils.validation import normalize_email

# Jira keys look like HELP-123. Validating before building a URL prevents someone
# from typing an "ID" such as "../../something" and altering the request path.
_TICKET_ID_RE = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")

# Jira groups every custom workflow status into one of three fixed "categories".
# Mapping the category (not the status name) means custom workflow names such as
# "Waiting for Vendor" still resolve correctly without any special-casing.
_CATEGORY_TO_STATUS = {
    "new": TicketStatus.OPEN,
    "indeterminate": TicketStatus.IN_PROGRESS,
    "done": TicketStatus.DONE,
}

_FIELDS = ["summary", "status", "labels", "created", "updated"]


class _TransientJiraError(Exception):
    """Internal marker for errors worth retrying (429 and 5xx)."""


def requester_label(email: str) -> str:
    """
    Build the Jira label that identifies a requester.

    Labels cannot contain spaces, and we want something predictable, so '@'
    becomes '_at_' and any other odd character becomes '_'. The same email
    always yields the same label, which is what makes the JQL lookup work.
    Public (no underscore) so tests can assert on it.
    """
    safe = re.sub(r"[^a-z0-9._-]", "_", normalize_email(email).replace("@", "_at_"))
    return f"req-{safe}"


def _adf(text: str) -> dict:
    """
    Wrap plain text in Atlassian Document Format (ADF).

    WHY: API v3 rejects plain-string descriptions; it requires this JSON tree.
    Each non-empty line becomes a paragraph (ADF forbids empty text nodes).
    """
    paragraphs = [
        {"type": "paragraph", "content": [{"type": "text", "text": line}]}
        for line in text.splitlines()
        if line.strip()
    ]
    return {"type": "doc", "version": 1, "content": paragraphs}


class JiraProvider(TicketProvider):
    """Creates and reads tickets in a single Jira Cloud project."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        """
        Args:
            client: optional pre-built httpx client. WHY: tests inject one with a
                fake transport so no real HTTP happens; production passes nothing.

        Raises:
            TicketProviderError: if required Jira settings are missing, so
            misconfiguration fails at startup rather than on the first user's ticket.
        """
        s = get_settings()
        token = s.jira_api_token.get_secret_value()
        if not (s.jira_base_url and s.jira_email and token):
            raise TicketProviderError(
                "Jira is not configured: set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN."
            )
        self._project = s.jira_project_key
        self._issue_type = s.jira_issue_type
        self._base_url = s.jira_base_url.rstrip("/")
        self._http = client or httpx.Client(
            base_url=self._base_url,
            auth=(s.jira_email, token),  # Jira Cloud: basic auth = email + API token
            headers={"Accept": "application/json"},
            timeout=15.0,                # never hang a chat turn on a slow Jira
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def create_ticket(self, request: TicketCreateRequest) -> Ticket:
        """
        Create a Jira issue and return it as a Ticket.

        The description is prefixed with the requester's email so the human agent
        sees who to contact. A second call fetches the issue because Jira's create
        response only contains the key, not the status or timestamps.
        """
        body = {
            "fields": {
                "project": {"key": self._project},
                "issuetype": {"name": self._issue_type},
                "summary": request.title,
                "description": _adf(
                    f"Requester: {request.requester_email}\n\n{request.description}"
                ),
                "labels": ["smartdesk", requester_label(request.requester_email)],
            }
        }
        created = self._request("POST", "/rest/api/3/issue", json=body)
        try:
            issue = self._request(
                "GET", f"/rest/api/3/issue/{created['key']}",
                params={"fields": ",".join(_FIELDS)},
            )
        except TicketProviderError:
            # The ticket EXISTS; only the read-back failed. Reporting failure here
            # would make the employee retry and create a duplicate.
            key = created["key"]
            return Ticket(
                id=key, title=request.title, status=TicketStatus.OPEN, raw_status="Open",
                url=f"{self._base_url}/browse/{key}", requester_email=request.requester_email,
            )
        return self._to_ticket(issue, request.requester_email)

    def get_ticket(self, ticket_id: str, requester_email: str) -> Ticket:
        """
        Fetch an issue by key, enforcing that it belongs to `requester_email`.

        A wrong-owner ticket raises TicketNotFoundError, the same as a missing
        one, so the response never confirms that someone else's ticket exists.
        """
        key = ticket_id.strip().upper()
        if not _TICKET_ID_RE.match(key):
            raise TicketNotFoundError(ticket_id)

        issue = self._request(
            "GET", f"/rest/api/3/issue/{key}", params={"fields": ",".join(_FIELDS)}
        )
        if requester_label(requester_email) not in issue["fields"].get("labels", []):
            raise TicketNotFoundError(ticket_id)
        return self._to_ticket(issue, requester_email)

    def find_tickets_by_email(self, requester_email: str, limit: int = 10) -> list[Ticket]:
        """
        List the requester's issues via JQL on their label, newest first.

        Uses POST /rest/api/3/search/jql (Atlassian replaced the older /search
        endpoint). The label is built from sanitized characters only, so it is
        safe to embed in the JQL string.
        """
        jql = (
            f'project = "{self._project}" AND labels = "{requester_label(requester_email)}" '
            "ORDER BY created DESC"
        )
        data = self._request(
            "POST",
            "/rest/api/3/search/jql",
            json={"jql": jql, "fields": _FIELDS, "maxResults": limit},
        )
        return [self._to_ticket(i, requester_email) for i in data.get("issues", [])]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            return self._request_with_retry(method, path, **kwargs)
        except _TransientJiraError as exc:
            raise TicketProviderError(str(exc)) from exc

    @retry(
        retry=retry_if_exception_type(_TransientJiraError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        reraise=True,
    )
    def _request_with_retry(self, method: str, path: str, **kwargs) -> dict:
        """
        Single choke point for every HTTP call to Jira.

        WHY CENTRALIZED: retry policy and error translation live in one place.
        - 429 / 5xx / network errors -> retried (transient), then surfaced as
          TicketProviderError.
        - 404 -> TicketNotFoundError (an expected user situation).
        - other 4xx (bad token, bad project key, wrong issue type) -> immediate
          TicketProviderError; retrying can't fix a config problem.
        """
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.TransportError as exc:  # timeouts, DNS, connection resets
            raise _TransientJiraError(str(exc)) from exc

        if (
            resp.status_code == 404
            and method.upper() == "GET"
            and path.startswith("/rest/api/3/issue/")
        ):
            raise TicketNotFoundError(path)
        if resp.status_code == 404:
            raise TicketProviderError(
                f"Jira endpoint or resource was not found ({method.upper()} {path})"
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _TransientJiraError(f"Jira returned {resp.status_code}")
        if resp.status_code >= 400:
            detail = self._error_detail(resp)
            message = f"Jira rejected the request ({resp.status_code})"
            if detail:
                message += f": {detail}"
            raise TicketProviderError(message)
        return resp.json()

    @staticmethod
    def _error_detail(resp: httpx.Response) -> str:
        """Extract Jira's validation messages without exposing arbitrary response bodies."""
        try:
            payload = resp.json()
        except ValueError:
            return ""
        if not isinstance(payload, dict):
            return ""

        messages = payload.get("errorMessages", [])
        errors = payload.get("errors", {})
        details = [str(message) for message in messages if message]
        if isinstance(errors, dict):
            details.extend(f"{field}: {message}" for field, message in errors.items())
        return "; ".join(details)[:1000]

    def _to_ticket(self, issue: dict, requester_email: str) -> Ticket:
        """
        Convert a raw Jira issue JSON into the neutral Ticket model.

        requester_email is passed in rather than parsed back out of the label,
        because the label encoding is one-way (lossy). Callers only reach here
        after the ownership/label check, so the value is already trusted.
        """
        f = issue["fields"]
        status = f["status"]
        return Ticket(
            id=issue["key"],
            title=f["summary"],
            status=_CATEGORY_TO_STATUS.get(
                status["statusCategory"]["key"], TicketStatus.IN_PROGRESS
            ),
            raw_status=status["name"],
            url=f"{self._base_url}/browse/{issue['key']}",
            requester_email=normalize_email(requester_email),
            created_at=datetime.fromisoformat(f["created"]),
            updated_at=datetime.fromisoformat(f["updated"]),
        )
    
