"""
Unit and Integration Test Suite for Agentic Self-Healing Actions Engine (#4115).
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.actions import router as actions_router
import backend.routers.actions as actions_module
from backend.services.agentic_action_engine import (
    AgenticActionEngine,
    PlaybookAuthorizationError,
    PlaybookValidationError,
)


class FakeSupabaseTable:
    def __init__(self, storage: list):
        self.storage = storage
        self._pending_update = None
        self._filter_id = None

    def insert(self, payload: dict):
        self.storage.append(payload)
        self._last = [payload]
        return self

    def update(self, updates: dict):
        self._pending_update = updates
        return self

    def select(self, *_args, **_kwargs):
        self._last = list(self.storage)
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def eq(self, key: str, value: str):
        self._filter_id = (key, value)
        return self

    def execute(self):
        if self._pending_update is not None:
            record = {"id": self._filter_id[1] if self._filter_id else "unknown"}
            record.update(self._pending_update)
            self.storage.append(record)
            self._last = [record]
            self._pending_update = None
        return type("Resp", (), {"data": getattr(self, "_last", list(self.storage))})()


class FakeSupabaseClient:
    def __init__(self):
        self.tables = {
            "action_execution_logs": [],
            "tickets": [],
        }

    def table(self, name: str) -> FakeSupabaseTable:
        if name not in self.tables:
            self.tables[name] = []
        return FakeSupabaseTable(self.tables[name])


def test_playbook_registry_and_schema_validation():
    engine = AgenticActionEngine(secret_key="test-secret")
    playbooks = engine.list_playbooks()
    names = {p["name"] for p in playbooks}
    assert {"password_reset", "vpn_cert_refresh", "dns_flush"}.issubset(names)

    # Valid parameters pass
    engine.validate_playbook_params("password_reset", {"user_email": "alice@example.com"})

    # Unknown playbook raises PlaybookValidationError
    with pytest.raises(PlaybookValidationError):
        engine.validate_playbook_params("non_existent_playbook", {})

    # Missing required parameter raises PlaybookValidationError
    with pytest.raises(PlaybookValidationError):
        engine.validate_playbook_params("vpn_cert_refresh", {"user_email": "alice@example.com"})

    # Invalid parameter type raises PlaybookValidationError
    with pytest.raises(PlaybookValidationError):
        engine.validate_playbook_params("dns_flush", {"hostname": 12345})


def test_hmac_authorization_and_audit_logging_on_failure():
    engine = AgenticActionEngine(secret_key="super-secret-key")
    fake_db = FakeSupabaseClient()
    params = {"user_email": "bob@example.com"}

    with pytest.raises(PlaybookAuthorizationError):
        engine.execute_playbook(
            ticket_id="ticket-101",
            playbook_name="password_reset",
            params=params,
            hmac_token="invalid-hmac-signature",
            supabase_client=fake_db,
        )

    assert len(fake_db.tables["action_execution_logs"]) == 1
    assert fake_db.tables["action_execution_logs"][0]["status"] == "unauthorized"
    assert len(fake_db.tables["tickets"]) == 0


def test_dry_run_does_not_resolve_ticket():
    engine = AgenticActionEngine(secret_key="super-secret-key")
    fake_db = FakeSupabaseClient()
    params = {"hostname": "internal.corp.local", "record_type": "AAAA"}
    token = engine.generate_hmac_token("ticket-202", "dns_flush", params)

    result = engine.execute_playbook(
        ticket_id="ticket-202",
        playbook_name="dns_flush",
        params=params,
        hmac_token=token,
        supabase_client=fake_db,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["status"] == "dry_run_success"
    assert result["ticket_resolved"] is False
    assert len(fake_db.tables["action_execution_logs"]) == 1
    assert len(fake_db.tables["tickets"]) == 0


def test_live_execution_deflects_ticket_and_writes_audit_log():
    engine = AgenticActionEngine(secret_key="super-secret-key")
    fake_db = FakeSupabaseClient()
    params = {"user_email": "carol@example.com", "device_id": "laptop-99"}
    token = engine.generate_hmac_token("ticket-303", "vpn_cert_refresh", params)

    result = engine.execute_playbook(
        ticket_id="ticket-303",
        playbook_name="vpn_cert_refresh",
        params=params,
        hmac_token=token,
        supabase_client=fake_db,
        dry_run=False,
        company_id="comp-1",
    )

    assert result["status"] == "succeeded"
    assert result["ticket_resolved"] is True
    assert "[EXECUTED]" in result["diagnostic_summary"]
    assert len(fake_db.tables["action_execution_logs"]) == 1
    assert fake_db.tables["action_execution_logs"][0]["company_id"] == "comp-1"
    assert len(fake_db.tables["tickets"]) == 1
    assert fake_db.tables["tickets"][0]["status"] == "resolved"
    assert "vpn_cert_refresh" in fake_db.tables["tickets"][0]["resolution_note"]


def test_fastapi_router_endpoints(monkeypatch):
    fake_db = FakeSupabaseClient()
    monkeypatch.setattr(actions_module, "supabase", fake_db)

    app = FastAPI()
    app.include_router(actions_router)
    client = TestClient(app)

    # 1. List playbooks
    res_list = client.get("/actions/playbooks")
    assert res_list.status_code == 200
    assert len(res_list.json()["playbooks"]) >= 3

    # 2. Execute playbook with valid HMAC
    params = {"user_email": "dave@example.com"}
    token = actions_module.agentic_action_engine.generate_hmac_token(
        "ticket-404", "password_reset", params
    )
    res_exec = client.post(
        "/actions/execute",
        json={
            "ticket_id": "ticket-404",
            "playbook_name": "password_reset",
            "parameters": params,
            "hmac_token": token,
            "dry_run": False,
        },
    )
    assert res_exec.status_code == 200
    body = res_exec.json()
    assert body["ticket_resolved"] is True
    assert body["status"] == "succeeded"

    # 3. Query audit logs
    res_logs = client.get("/actions/logs?ticket_id=ticket-404")
    assert res_logs.status_code == 200
    assert res_logs.json()["count"] >= 1
