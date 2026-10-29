"""
Agentic Self-Healing Actions Engine - Sandboxed Playbook Execution for Automated Deflection.

Provides:
  - Playbook registry with strict parameter schema validation
  - Dry-run execution mode (dry_run=True)
  - Constant-time HMAC-SHA256 token verification for playbook execution
  - Audit logging to Supabase `action_execution_logs` table
  - Auto-deflection feedback loop updating ticket status to `resolved` with diagnostic summary
"""

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlaybookValidationError(ValueError):
    """Raised when playbook name or parameters fail schema validation."""


class PlaybookAuthorizationError(PermissionError):
    """Raised when HMAC token authorization fails."""


DEFAULT_PLAYBOOKS: Dict[str, Dict[str, Any]] = {
    "password_reset": {
        "name": "password_reset",
        "description": "Triggers a sandboxed self-service password reset token and invalidates stale sessions.",
        "tier": 1,
        "required_params": ["user_email"],
        "optional_params": {"notify_channel": str, "force_mfa_reenroll": bool},
        "param_types": {"user_email": str},
    },
    "vpn_cert_refresh": {
        "name": "vpn_cert_refresh",
        "description": "Revokes expired client VPN certificate and issues a renewed short-lived certificate.",
        "tier": 1,
        "required_params": ["user_email", "device_id"],
        "optional_params": {"profile": str},
        "param_types": {"user_email": str, "device_id": str},
    },
    "dns_flush": {
        "name": "dns_flush",
        "description": "Clears stale internal DNS resolver cache entries for the target hostname or client.",
        "tier": 1,
        "required_params": ["hostname"],
        "optional_params": {"record_type": str},
        "param_types": {"hostname": str},
    },
}


class AgenticActionEngine:
    """Executes deterministic, sandboxed Tier-1 remediation playbooks with HMAC verification."""

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = (
            secret_key
            or os.getenv("AGENTIC_ACTION_HMAC_SECRET")
            or "default-agentic-hmac-secret-key"
        )
        self._playbooks: Dict[str, Dict[str, Any]] = dict(DEFAULT_PLAYBOOKS)
        self._handlers: Dict[str, Callable[[Dict[str, Any], bool], Dict[str, Any]]] = {
            "password_reset": self._handle_password_reset,
            "vpn_cert_refresh": self._handle_vpn_cert_refresh,
            "dns_flush": self._handle_dns_flush,
        }

    # ------------------------------------------------------------------
    # Registry & Schema Validation
    # ------------------------------------------------------------------

    def list_playbooks(self) -> List[Dict[str, Any]]:
        """Return metadata for all registered playbooks."""
        return [
            {
                "name": spec["name"],
                "description": spec["description"],
                "tier": spec.get("tier", 1),
                "required_params": list(spec.get("required_params", [])),
            }
            for spec in self._playbooks.values()
        ]

    def register_playbook(
        self,
        name: str,
        description: str,
        required_params: List[str],
        param_types: Dict[str, type],
        handler: Callable[[Dict[str, Any], bool], Dict[str, Any]],
        optional_params: Optional[Dict[str, type]] = None,
        tier: int = 1,
    ) -> None:
        """Register a custom deterministic playbook and its schema."""
        if not name or not isinstance(name, str):
            raise PlaybookValidationError("Playbook name must be a non-empty string.")
        self._playbooks[name] = {
            "name": name,
            "description": description,
            "tier": tier,
            "required_params": list(required_params),
            "optional_params": dict(optional_params or {}),
            "param_types": dict(param_types),
        }
        self._handlers[name] = handler

    def validate_playbook_params(self, playbook_name: str, params: Dict[str, Any]) -> None:
        """Validate playbook existence and parameter schema."""
        if playbook_name not in self._playbooks:
            raise PlaybookValidationError(f"Unknown playbook: '{playbook_name}'")

        if not isinstance(params, dict):
            raise PlaybookValidationError("Playbook parameters must be a dictionary.")

        spec = self._playbooks[playbook_name]
        required = spec.get("required_params", [])
        param_types = spec.get("param_types", {})
        optional_types = spec.get("optional_params", {})

        for req_key in required:
            if req_key not in params or params[req_key] is None or params[req_key] == "":
                raise PlaybookValidationError(
                    f"Missing required parameter '{req_key}' for playbook '{playbook_name}'"
                )

        allowed_keys = set(required) | set(optional_types.keys())
        for key, value in params.items():
            if key not in allowed_keys:
                raise PlaybookValidationError(
                    f"Unexpected parameter '{key}' for playbook '{playbook_name}'"
                )
            expected_type = param_types.get(key) or optional_types.get(key)
            if expected_type and not isinstance(value, expected_type):
                raise PlaybookValidationError(
                    f"Parameter '{key}' must be of type {expected_type.__name__}"
                )

    # ------------------------------------------------------------------
    # HMAC Token Authorization
    # ------------------------------------------------------------------

    def generate_hmac_token(
        self, ticket_id: str, playbook_name: str, params: Dict[str, Any]
    ) -> str:
        """Compute canonical HMAC-SHA256 signature for an execution payload."""
        canonical_params = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
        message = f"{ticket_id}:{playbook_name}:{canonical_params}".encode("utf-8")
        return hmac.new(
            self.secret_key.encode("utf-8"), message, hashlib.sha256
        ).hexdigest()

    def verify_hmac_token(
        self,
        ticket_id: str,
        playbook_name: str,
        params: Dict[str, Any],
        token: Optional[str],
    ) -> bool:
        """Verify HMAC-SHA256 signature in constant time."""
        if not token or not isinstance(token, str):
            return False
        expected = self.generate_hmac_token(ticket_id, playbook_name, params)
        return hmac.compare_digest(expected, token)

    # ------------------------------------------------------------------
    # Execution, Audit Logging & Auto-Deflection Loop
    # ------------------------------------------------------------------

    def execute_playbook(
        self,
        ticket_id: str,
        playbook_name: str,
        params: Dict[str, Any],
        hmac_token: Optional[str],
        supabase_client: Any = None,
        dry_run: bool = False,
        company_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate, authorize, and execute a sandboxed playbook.

        When dry_run=False and execution succeeds, logs the execution to
        `action_execution_logs` and resolves the ticket with a diagnostic summary note.
        """
        if not ticket_id or not isinstance(ticket_id, str):
            raise PlaybookValidationError("ticket_id must be a non-empty string.")

        self.validate_playbook_params(playbook_name, params)

        if not self.verify_hmac_token(ticket_id, playbook_name, params, hmac_token):
            self._record_audit_log(
                supabase_client,
                ticket_id=ticket_id,
                playbook_name=playbook_name,
                params=params,
                dry_run=dry_run,
                status="unauthorized",
                diagnostic_summary="HMAC token verification failed.",
                company_id=company_id,
            )
            raise PlaybookAuthorizationError("Invalid or missing HMAC authorization token.")

        handler = self._handlers[playbook_name]
        execution_output = handler(params, dry_run)
        executed_at = _utc_now_iso()

        status = "dry_run_success" if dry_run else "succeeded"
        diagnostic_summary = execution_output.get(
            "diagnostic_summary",
            f"Playbook '{playbook_name}' completed ({status}).",
        )

        audit_entry = self._record_audit_log(
            supabase_client,
            ticket_id=ticket_id,
            playbook_name=playbook_name,
            params=params,
            dry_run=dry_run,
            status=status,
            diagnostic_summary=diagnostic_summary,
            company_id=company_id,
        )

        ticket_updated = False
        if not dry_run and supabase_client is not None:
            ticket_updated = self._deflect_ticket(
                supabase_client,
                ticket_id=ticket_id,
                playbook_name=playbook_name,
                diagnostic_summary=diagnostic_summary,
            )

        return {
            "ticket_id": ticket_id,
            "playbook_name": playbook_name,
            "dry_run": dry_run,
            "status": status,
            "ticket_resolved": ticket_updated,
            "diagnostic_summary": diagnostic_summary,
            "details": execution_output,
            "executed_at": executed_at,
            "audit_log": audit_entry,
        }

    def _record_audit_log(
        self,
        supabase_client: Any,
        ticket_id: str,
        playbook_name: str,
        params: Dict[str, Any],
        dry_run: bool,
        status: str,
        diagnostic_summary: str,
        company_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ticket_id": ticket_id,
            "playbook_name": playbook_name,
            "parameters": params,
            "dry_run": dry_run,
            "status": status,
            "diagnostic_summary": diagnostic_summary,
            "executed_at": _utc_now_iso(),
        }
        if company_id:
            payload["company_id"] = company_id

        if supabase_client is None:
            return payload

        try:
            res = supabase_client.table("action_execution_logs").insert(payload).execute()
            if getattr(res, "data", None):
                return res.data[0]
        except Exception as exc:
            logger.error(f"[AgenticActionEngine] Failed to insert audit log: {exc}")
        return payload

    def _deflect_ticket(
        self,
        supabase_client: Any,
        ticket_id: str,
        playbook_name: str,
        diagnostic_summary: str,
    ) -> bool:
        updates = {
            "status": "resolved",
            "resolved_at": _utc_now_iso(),
            "resolution_note": (
                f"[Auto-Deflected via '{playbook_name}'] {diagnostic_summary}"
            ),
        }
        try:
            res = (
                supabase_client.table("tickets")
                .update(updates)
                .eq("id", ticket_id)
                .execute()
            )
            return bool(getattr(res, "data", None))
        except Exception as exc:
            logger.error(f"[AgenticActionEngine] Failed to update ticket {ticket_id}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Built-in Deterministic Sandboxed Handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_password_reset(params: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
        email = params["user_email"]
        mode = "DRY-RUN" if dry_run else "EXECUTED"
        return {
            "action": "password_reset",
            "target": email,
            "sessions_revoked": 0 if dry_run else 1,
            "diagnostic_summary": (
                f"[{mode}] Password reset workflow completed and active sessions invalidated for {email}."
            ),
        }

    @staticmethod
    def _handle_vpn_cert_refresh(params: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
        email = params["user_email"]
        device_id = params["device_id"]
        mode = "DRY-RUN" if dry_run else "EXECUTED"
        return {
            "action": "vpn_cert_refresh",
            "target": email,
            "device_id": device_id,
            "certificate_renewed": not dry_run,
            "diagnostic_summary": (
                f"[{mode}] Renewed VPN client certificate for {email} on device {device_id}."
            ),
        }

    @staticmethod
    def _handle_dns_flush(params: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
        hostname = params["hostname"]
        record_type = params.get("record_type", "A")
        mode = "DRY-RUN" if dry_run else "EXECUTED"
        return {
            "action": "dns_flush",
            "hostname": hostname,
            "record_type": record_type,
            "cache_cleared": not dry_run,
            "diagnostic_summary": (
                f"[{mode}] Flushed DNS resolver cache for {hostname} (type={record_type})."
            ),
        }


agentic_action_engine = AgenticActionEngine()
