"""
Unit and Integration Test Suite for Predictive SLA Breach Forecasting (#4116).
"""

import pytest
from backend.services.sla_forecasting_service import (
    HIGH_RISK_TAG,
    SLAForecastingService,
)


class FakeSupabaseTable:
    """In-memory mock representing a Supabase table query builder for SLA forecasting."""

    def __init__(self, rows: list):
        """Initialize the mock table with a list of ticket rows."""
        self.rows = rows
        self._updates = None
        self._target_id = None
        self._filters = {}

    def select(self, *_args, **_kwargs):
        """Select all rows in the mock table."""
        return self

    def eq(self, key: str, value: str):
        """Record an equality filter on the ticket query."""
        if key == "id":
            self._target_id = value
        self._filters[key] = value
        return self

    def neq(self, *_args, **_kwargs):
        """No-op inequality filter for mock queries."""
        return self

    def update(self, updates: dict):
        """Stage a dictionary of updates to apply on execute."""
        self._updates = updates
        return self

    def execute(self):
        """Apply any staged updates and return a filtered response object."""
        if self._updates is not None and self._target_id is not None:
            for r in self.rows:
                if str(r.get("id")) == str(self._target_id):
                    r.update(self._updates)
            self._updates = None
        matched = list(self.rows)
        for k, v in self._filters.items():
            matched = [r for r in matched if str(r.get(k)) == str(v)]
        return type("Resp", (), {"data": matched})()


class FakeSupabaseClient:
    """In-memory mock representing the Supabase client for SLA forecasting tests."""

    def __init__(self, tickets: list):
        """Initialize the client with a list of mock tickets."""
        self.tickets = tickets

    def table(self, _name: str) -> FakeSupabaseTable:
        """Return a FakeSupabaseTable bound to the ticket list."""
        return FakeSupabaseTable(self.tickets)


def test_breach_probability_bounds_and_monotonicity():
    """Verify P(breach) stays in [0.0, 1.0] and increases monotonically with workload and complexity."""
    svc = SLAForecastingService()

    low_prob = svc.calculate_breach_probability(
        queue_depth=1,
        description="Reset user email preference",
        historical_velocity=6.0,
    )
    high_prob = svc.calculate_breach_probability(
        queue_depth=35,
        description="Critical production database outage and multi-tenant corruption breach",
        historical_velocity=0.3,
    )

    assert 0.0 <= low_prob <= 1.0
    assert 0.0 <= high_prob <= 1.0
    assert high_prob > 0.75
    assert low_prob < 0.40
    assert high_prob > low_prob


def test_high_risk_threshold_tags_ticket_and_triggers_webhook(monkeypatch):
    """Verify tickets with P(breach) > 0.75 receive risk:high-breach tag and trigger webhook notification."""
    svc = SLAForecastingService(webhook_url="https://hooks.example.com/sla")
    webhook_calls = []

    def fake_notify(ticket, probability, webhook_url=""):
        """Capture webhook notification parameters during testing."""
        webhook_calls.append((ticket["id"], probability, webhook_url))
        return True

    monkeypatch.setattr(svc, "notify_on_duty_leads", fake_notify)

    ticket = {
        "id": "t-900",
        "subject": "Critical production database outage",
        "description": "Multi-tenant authentication deadlock and security breach",
        "queue_depth": 40,
        "historical_velocity": 0.25,
        "tags": ["tier-1"],
    }
    fake_db = FakeSupabaseClient([ticket])

    result = svc.evaluate_ticket(ticket, supabase_client=fake_db)
    assert result["breach_probability"] > 0.75
    assert result["high_risk"] is True
    assert result["risk_tier"] == "high"
    assert HIGH_RISK_TAG in result["tags"]
    assert result["webhook_notified"] is True
    assert len(webhook_calls) == 1


def test_company_risk_distribution(monkeypatch):
    """Verify get_company_risk_distribution aggregates buckets without sending notifications."""
    svc = SLAForecastingService(webhook_url="https://hooks.example.com/sla")
    webhook_calls = []
    monkeypatch.setattr(
        svc,
        "notify_on_duty_leads",
        lambda ticket, probability, webhook_url="": webhook_calls.append(ticket["id"]) or True,
    )
    tickets = [
        {
            "id": "t-1",
            "company_id": "acme",
            "subject": "Minor UI typo",
            "description": "Label alignment",
            "queue_depth": None,
            "historical_velocity": None,
        },
        {
            "id": "t-2",
            "company_id": "acme",
            "subject": "Critical production outage",
            "description": "Database corruption and security breach timeout",
            "queue_depth": 50,
            "historical_velocity": 0.2,
        },
    ]
    fake_db = FakeSupabaseClient(tickets)
    summary = svc.get_company_risk_distribution(fake_db, company_id="acme")

    assert summary["company_id"] == "acme"
    assert summary["total_open_tickets"] == 2
    assert summary["risk_distribution"]["high"] == 1
    assert summary["risk_distribution"]["low"] == 1
    assert len(summary["high_risk_tickets"]) == 1
    assert len(webhook_calls) == 0
