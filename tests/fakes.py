"""Shared test doubles (no network, no API key)."""

from smartdesk.llm.schemas import FollowUpDecision, IntentResult, TicketSummary


class ScriptedLLM:
    """
    Stand-in for LLMClient whose answers are set by the test.

    `intent` is what the router "decides" for fresh messages; `decision` is how
    it classifies ambiguous replies to a pending question. Tests change them
    between turns. `parsed` records which schemas were requested, so tests can
    assert that an LLM call was (or wasn't) needed.
    """

    def __init__(self, intent="kb_query", decision="other"):
        self.intent, self.decision = intent, decision
        self.summary = TicketSummary(
            title="Cannot connect to VPN",
            problem="VPN shows error 809 on a Mac.",
            reasoning="scripted",
        )
        self.parsed: list[str] = []

    def parse(self, messages, schema):
        self.parsed.append(schema.__name__)
        if schema is IntentResult:
            return IntentResult(intent=self.intent, reasoning="scripted")
        if schema is FollowUpDecision:
            return FollowUpDecision(decision=self.decision, reasoning="scripted")
        if schema is TicketSummary:
            return self.summary
        raise AssertionError(f"Unexpected schema: {schema}")

    def chat(self, messages, temperature=0.2):
        return "Hi! I can answer IT and HR questions, raise tickets and check ticket status."

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]
    