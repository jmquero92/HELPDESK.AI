"""
Predictive SLA Breach Forecasting & Dynamic Capacity Balancing Engine (#4116).

Provides:
  - calculate_breach_probability() -> P(breach) in [0.0, 1.0] based on:
      1. Current technician active ticket queue depth
      2. Semantic complexity of ticket description
      3. Department historical resolution velocity
  - evaluate_ticket() -> auto-tags ticket with `risk:high-breach` and notifies
    on-duty leads via webhook when P(breach) > 0.75
  - get_company_risk_distribution() -> powers GET /internal/sla/forecast?company_id=...
"""

import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import requests as _requests_lib
except ImportError:
    _requests_lib = None  # type: ignore

logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD = 0.75
HIGH_RISK_TAG = "risk:high-breach"

COMPLEXITY_KEYWORDS = {
    "outage": 0.25,
    "production": 0.20,
    "critical": 0.20,
    "database": 0.15,
    "corruption": 0.25,
    "security": 0.20,
    "breach": 0.25,
    "sso": 0.12,
    "authentication": 0.12,
    "latency": 0.12,
    "timeout": 0.12,
    "deadlock": 0.20,
    "migration": 0.15,
    "multi-tenant": 0.18,
}


def _utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SLAForecastingService:
    """Proactively forecasts SLA breach probability at ticket creation and balances capacity."""

    def __init__(self, webhook_url: Optional[str] = None, threshold: float = HIGH_RISK_THRESHOLD):
        """Initialize the SLA forecasting service with webhook configuration and threshold."""
        self.webhook_url = (
            webhook_url
            or os.getenv("SLA_FORECAST_WEBHOOK_URL")
            or os.getenv("SLACK_WEBHOOK_URL")
            or ""
        )
        self.threshold = threshold

    @staticmethod
    def compute_semantic_complexity(description: str) -> float:
        """
        Estimate normalized semantic complexity in [0.0, 1.0] from ticket text length,
        technical tokens, and high-severity domain indicators.
        """
        if not description or not description.strip():
            return 0.1

        text = description.lower()
        words = re.findall(r"[a-z0-9_-]+", text)
        word_count = len(words)

        length_factor = min(1.0, word_count / 120.0) * 0.35
        keyword_score = 0.0
        for kw, weight in COMPLEXITY_KEYWORDS.items():
            if kw in text:
                keyword_score += weight

        score = 0.15 + length_factor + min(0.55, keyword_score)
        return round(max(0.0, min(1.0, score)), 4)

    def calculate_breach_probability(
        self,
        queue_depth: int,
        description: str,
        historical_velocity: float,
        complexity_override: Optional[float] = None,
    ) -> float:
        """
        Calculate SLA breach probability P(breach) in [0.0, 1.0].

        Args:
            queue_depth: Active open tickets assigned to the technician/team (>= 0).
            description: Ticket subject + body text.
            historical_velocity: Average tickets resolved per hour by department (> 0).
            complexity_override: Optional precomputed semantic complexity in [0.0, 1.0].
        """
        safe_queue = max(0, int(queue_depth))
        safe_velocity = max(0.1, float(historical_velocity))
        complexity = (
            float(complexity_override)
            if complexity_override is not None
            else self.compute_semantic_complexity(description)
        )
        complexity = max(0.0, min(1.0, complexity))

        # Queue pressure ratio relative to department velocity
        workload_ratio = safe_queue / (safe_velocity * 4.0)
        queue_component = min(1.0, 1.0 - math.exp(-0.55 * workload_ratio))

        # Velocity drag (lower velocity -> higher risk)
        velocity_drag = min(1.0, 1.0 / (0.6 + safe_velocity * 0.4))

        raw_score = (0.45 * queue_component) + (0.35 * complexity) + (0.20 * velocity_drag)
        return round(max(0.0, min(1.0, raw_score)), 4)

    @staticmethod
    def _derive_queue_depth(ticket: Dict[str, Any], supabase_client: Any = None) -> int:
        """Derive active technician ticket queue depth from ticket payload or repository."""
        raw_qd = ticket.get("queue_depth")
        if raw_qd is not None:
            return max(0, int(raw_qd))

        assignee = ticket.get("assigned_to") or ticket.get("technician_id")
        if supabase_client is not None and assignee:
            try:
                res = (
                    supabase_client.table("tickets")
                    .select("id")
                    .eq("assigned_to", assignee)
                    .neq("status", "resolved")
                    .neq("status", "closed")
                    .execute()
                )
                rows = getattr(res, "data", None) or []
                return len(rows)
            except Exception as exc:
                logger.warning(f"[SLAForecast] Failed to derive queue depth for {assignee}: {exc}")
        return 0

    @staticmethod
    def _derive_historical_velocity(ticket: Dict[str, Any], supabase_client: Any = None) -> float:
        """Derive department historical resolution velocity from ticket payload or repository."""
        raw_hv = ticket.get("historical_velocity")
        if raw_hv is not None:
            return max(0.1, float(raw_hv))

        dept = ticket.get("department") or ticket.get("department_id") or ticket.get("category")
        if supabase_client is not None and dept:
            try:
                res = (
                    supabase_client.table("tickets")
                    .select("id")
                    .eq("department", dept)
                    .eq("status", "resolved")
                    .execute()
                )
                rows = getattr(res, "data", None) or []
                if rows:
                    return max(0.1, round(len(rows) / 24.0, 4))
            except Exception as exc:
                logger.warning(f"[SLAForecast] Failed to derive velocity for {dept}: {exc}")
        return 2.0

    def evaluate_ticket(
        self,
        ticket: Dict[str, Any],
        supabase_client: Any = None,
        webhook_url: Optional[str] = None,
        notify: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a ticket's SLA breach probability, auto-tag with `risk:high-breach`
        and optionally notify on-duty leads via webhook when P(breach) > 0.75.
        """
        ticket_id = str(ticket.get("id", "unknown"))
        description = f"{ticket.get('subject', '')} {ticket.get('description', '')}".strip()
        queue_depth = self._derive_queue_depth(ticket, supabase_client=supabase_client)
        velocity = self._derive_historical_velocity(ticket, supabase_client=supabase_client)
        complexity_override = ticket.get("semantic_complexity")

        prob = self.calculate_breach_probability(
            queue_depth=queue_depth,
            description=description,
            historical_velocity=velocity,
            complexity_override=complexity_override,
        )

        is_high_risk = prob > self.threshold
        tags: List[str] = list(ticket.get("tags") or [])
        if is_high_risk and HIGH_RISK_TAG not in tags:
            tags.append(HIGH_RISK_TAG)

        webhook_sent = False
        if is_high_risk:
            if supabase_client is not None and ticket.get("id") and notify:
                try:
                    (
                        supabase_client.table("tickets")
                        .update(
                            {
                                "tags": tags,
                                "sla_breach_probability": prob,
                                "forecasted_at": _utc_now_iso(),
                            }
                        )
                        .eq("id", ticket_id)
                        .execute()
                    )
                except Exception as exc:
                    logger.error(f"[SLAForecast] Failed to tag ticket {ticket_id}: {exc}")

            if notify:
                webhook_sent = self.notify_on_duty_leads(
                    ticket=ticket,
                    probability=prob,
                    webhook_url=webhook_url or self.webhook_url,
                )

        risk_tier = "high" if is_high_risk else ("medium" if prob >= 0.4 else "low")
        return {
            "ticket_id": ticket_id,
            "breach_probability": prob,
            "risk_tier": risk_tier,
            "high_risk": is_high_risk,
            "tags": tags,
            "webhook_notified": webhook_sent,
        }

    def notify_on_duty_leads(
        self,
        ticket: Dict[str, Any],
        probability: float,
        webhook_url: str = "",
    ) -> bool:
        """Send an asynchronous webhook notification to on-duty engineering leads."""
        target_url = webhook_url or self.webhook_url
        if not target_url or _requests_lib is None:
            return False

        payload = {
            "text": (
                f":warning: *High SLA Breach Risk Forecasted* (`{HIGH_RISK_TAG}`)\n"
                f"• *Ticket ID:* `{ticket.get('id', 'N/A')}`\n"
                f"• *Subject:* {ticket.get('subject', 'Untitled')}\n"
                f"• *P(breach):* `{probability:.2f}` (threshold > {self.threshold})"
            )
        }
        try:
            resp = _requests_lib.post(target_url, json=payload, timeout=5)
            return 200 <= resp.status_code < 300
        except Exception as exc:
            logger.error(f"[SLAForecast] Webhook notification error: {exc}")
            return False

    def get_company_risk_distribution(
        self,
        supabase_client: Any,
        company_id: str,
    ) -> Dict[str, Any]:
        """
        Compute real-time SLA breach risk distribution for a company's open tickets.
        Powers `GET /internal/sla/forecast?company_id=...`.
        """
        tickets: List[Dict[str, Any]] = []
        if supabase_client is not None:
            try:
                res = (
                    supabase_client.table("tickets")
                    .select("*")
                    .eq("company_id", company_id)
                    .neq("status", "resolved")
                    .neq("status", "closed")
                    .execute()
                )
                tickets = getattr(res, "data", None) or []
            except Exception as exc:
                logger.error(f"[SLAForecast] Failed to fetch tickets for {company_id}: {exc}")

        distribution = {"low": 0, "medium": 0, "high": 0}
        evaluated: List[Dict[str, Any]] = []
        for t in tickets:
            ev = self.evaluate_ticket(t, supabase_client=supabase_client, notify=False)
            distribution[ev["risk_tier"]] += 1
            evaluated.append(ev)

        total = len(evaluated)
        avg_prob = (
            round(sum(item["breach_probability"] for item in evaluated) / total, 4)
            if total > 0
            else 0.0
        )
        return {
            "company_id": company_id,
            "total_open_tickets": total,
            "average_breach_probability": avg_prob,
            "risk_distribution": distribution,
            "high_risk_tickets": [e for e in evaluated if e["high_risk"]],
        }


sla_forecasting_service = SLAForecastingService()
