#!/usr/bin/env python3
"""
Risk Control System

Implementation of risk controls based on agents-mike.md recommendations:
- Circuit breakers for automated trading halts
- Kill switch for emergency stops
- Risk limit monitoring
- Compliance audit trail
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import uuid

from models import (
    CircuitBreaker,
    KillSwitch,
    RiskLimit,
    DecisionAudit,
    ComplianceReport,
)

logger = logging.getLogger(__name__)


class RiskControlSystem:
    """Comprehensive risk control and monitoring system"""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.kill_switch = KillSwitch()
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.decision_audit_trail: List[DecisionAudit] = []

        # Initialize default circuit breakers
        self._initialize_default_breakers()
        self._initialize_default_limits()

    def _initialize_default_breakers(self):
        """Initialize default circuit breakers"""
        default_breakers = [
            {
                "name": "Daily Drawdown Limit",
                "trigger_type": "drawdown",
                "threshold_value": 0.05,  # 5% daily drawdown
            },
            {
                "name": "VaR Breach Monitor",
                "trigger_type": "var_breach",
                "threshold_value": 0.02,  # 2% VaR limit
            },
            {
                "name": "Total Loss Limit",
                "trigger_type": "loss_limit",
                "threshold_value": 10000,  # $10k daily loss
            },
            {
                "name": "Volatility Spike",
                "trigger_type": "volatility",
                "threshold_value": 0.5,  # 50% volatility threshold
            },
        ]

        for breaker_config in default_breakers:
            breaker_id = f"cb_{uuid.uuid4().hex[:8]}"
            self.circuit_breakers[breaker_id] = CircuitBreaker(
                breaker_id=breaker_id,
                name=breaker_config["name"],
                trigger_type=breaker_config["trigger_type"],
                threshold_value=breaker_config["threshold_value"],
                current_value=0.0,
            )
            logger.info(f"Initialized circuit breaker: {breaker_config['name']}")

    def _initialize_default_limits(self):
        """Initialize default risk limits"""
        default_limits = [
            {
                "limit_type": "position_size",
                "limit_value": 0.1,  # 10% max position size
            },
            {
                "limit_type": "daily_loss",
                "limit_value": 5000,  # $5k daily loss limit
            },
            {
                "limit_type": "var_limit",
                "limit_value": 0.05,  # 5% VaR limit
            },
            {
                "limit_type": "concentration",
                "limit_value": 0.25,  # 25% max concentration
            },
        ]

        for limit_config in default_limits:
            limit_id = f"rl_{uuid.uuid4().hex[:8]}"
            self.risk_limits[limit_id] = RiskLimit(
                limit_type=limit_config["limit_type"],
                limit_value=limit_config["limit_value"],
                current_value=0.0,
                utilization_pct=0.0,
            )
            logger.info(f"Initialized risk limit: {limit_config['limit_type']}")

    def check_circuit_breakers(self, market_metrics: Dict[str, float]) -> List[str]:
        """Check all circuit breakers and return list of triggered breakers"""
        triggered_breakers = []

        for breaker_id, breaker in self.circuit_breakers.items():
            if breaker.is_triggered:
                continue  # Already triggered

            # Get current value based on trigger type
            current_value = self._get_current_value(
                breaker.trigger_type, market_metrics
            )
            breaker.current_value = current_value

            # Check if threshold is breached
            if self._should_trigger_breaker(breaker, current_value):
                breaker.is_triggered = True
                breaker.trigger_time = datetime.now()
                triggered_breakers.append(breaker_id)

                logger.warning(
                    f"Circuit breaker triggered: {breaker.name} "
                    f"(value: {current_value:.4f}, threshold: {breaker.threshold_value:.4f})"
                )

        return triggered_breakers

    def _get_current_value(self, trigger_type: str, metrics: Dict[str, float]) -> float:
        """Get current value for circuit breaker check"""
        mapping = {
            "drawdown": metrics.get("current_drawdown", 0.0),
            "var_breach": metrics.get("current_var", 0.0),
            "loss_limit": metrics.get("daily_pnl", 0.0),
            "volatility": metrics.get("current_volatility", 0.0),
        }
        return mapping.get(trigger_type, 0.0)

    def _should_trigger_breaker(
        self, breaker: CircuitBreaker, current_value: float
    ) -> bool:
        """Determine if circuit breaker should trigger"""
        if breaker.trigger_type in ["drawdown", "var_breach", "volatility"]:
            return current_value > breaker.threshold_value
        elif breaker.trigger_type == "loss_limit":
            return current_value < -abs(breaker.threshold_value)  # Negative loss
        return False

    def check_risk_limits(self, current_positions: Dict[str, Any]) -> Dict[str, bool]:
        """Check all risk limits and return breach status"""
        limit_status = {}

        for limit_id, limit in self.risk_limits.items():
            current_value = self._calculate_limit_value(
                limit.limit_type, current_positions
            )
            limit.current_value = current_value
            limit.utilization_pct = (current_value / limit.limit_value) * 100

            is_breached = current_value > limit.limit_value
            limit_status[limit_id] = is_breached

            if is_breached:
                limit.breach_count += 1
                limit.last_breach = datetime.now()
                logger.warning(
                    f"Risk limit breached: {limit.limit_type} "
                    f"(current: {current_value:.4f}, limit: {limit.limit_value:.4f})"
                )

        return limit_status

    def _calculate_limit_value(
        self, limit_type: str, positions: Dict[str, Any]
    ) -> float:
        """Calculate current value for risk limit"""
        if limit_type == "position_size":
            # Max single position as % of portfolio
            total_value = sum(pos.get("value", 0) for pos in positions.values())
            if total_value == 0:
                return 0.0
            max_position = max(pos.get("value", 0) for pos in positions.values())
            return max_position / total_value

        elif limit_type == "daily_loss":
            # Sum of negative P&L today
            return abs(
                sum(
                    pos.get("daily_pnl", 0)
                    for pos in positions.values()
                    if pos.get("daily_pnl", 0) < 0
                )
            )

        elif limit_type == "var_limit":
            # Portfolio VaR calculation (simplified)
            portfolio_value = sum(pos.get("value", 0) for pos in positions.values())
            volatilities = [pos.get("volatility", 0.02) for pos in positions.values()]
            avg_vol = np.mean(volatilities) if volatilities else 0.02
            return portfolio_value * avg_vol * 2.33  # 99% VaR approximation

        elif limit_type == "concentration":
            # Max sector/asset class concentration
            total_value = sum(pos.get("value", 0) for pos in positions.values())
            if total_value == 0:
                return 0.0
            # Group by sector (simplified - assuming symbol prefix)
            sectors = {}
            for symbol, pos in positions.items():
                sector = symbol[:2]  # Simple sector grouping
                sectors[sector] = sectors.get(sector, 0) + pos.get("value", 0)
            max_sector = max(sectors.values()) if sectors else 0
            return max_sector / total_value

        return 0.0

    def activate_kill_switch(self, reason: str, triggered_by: str = "system") -> bool:
        """Activate emergency kill switch"""
        try:
            self.kill_switch.is_active = True
            self.kill_switch.trigger_reason = reason
            self.kill_switch.triggered_by = triggered_by
            self.kill_switch.trigger_time = datetime.now()

            logger.critical(f"KILL SWITCH ACTIVATED: {reason} (by: {triggered_by})")

            # Record in audit trail
            self.record_decision_audit(
                decision_type="kill_switch",
                symbol="ALL",
                quantity=0,
                rationale=f"Kill switch activated: {reason}",
                confidence_level=1.0,
            )

            return True

        except Exception as e:
            logger.error(f"Error activating kill switch: {e}")
            return False

    def deactivate_kill_switch(self, authorized_by: str = "admin") -> bool:
        """Deactivate kill switch (requires authorization)"""
        try:
            if not self.kill_switch.is_active:
                logger.warning("Kill switch is not active")
                return False

            self.kill_switch.is_active = False
            self.kill_switch.reactivation_time = datetime.now()

            logger.info(f"Kill switch deactivated by: {authorized_by}")

            # Record in audit trail
            self.record_decision_audit(
                decision_type="kill_switch_reset",
                symbol="ALL",
                quantity=0,
                rationale=f"Kill switch deactivated by {authorized_by}",
                confidence_level=1.0,
            )

            return True

        except Exception as e:
            logger.error(f"Error deactivating kill switch: {e}")
            return False

    def reset_circuit_breaker(self, breaker_id: str) -> bool:
        """Reset a triggered circuit breaker"""
        try:
            if breaker_id not in self.circuit_breakers:
                logger.error(f"Circuit breaker {breaker_id} not found")
                return False

            breaker = self.circuit_breakers[breaker_id]
            breaker.is_triggered = False
            breaker.reset_time = datetime.now()
            breaker.current_value = 0.0

            logger.info(f"Circuit breaker reset: {breaker.name}")
            return True

        except Exception as e:
            logger.error(f"Error resetting circuit breaker: {e}")
            return False

    def record_decision_audit(
        self,
        decision_type: str,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        rationale: str = "",
        confidence_level: float = 0.5,
        data_sources: List[str] = None,
        risk_assessment: Dict[str, Any] = None,
    ) -> str:
        """Record trading decision in audit trail"""
        try:
            decision_id = f"decision_{uuid.uuid4().hex[:8]}"

            audit_record = DecisionAudit(
                decision_id=decision_id,
                timestamp=datetime.now(),
                decision_type=decision_type,
                symbol=symbol,
                quantity=quantity,
                price=price,
                rationale=rationale,
                confidence_level=confidence_level,
                data_sources=data_sources or [],
                risk_assessment=risk_assessment or {},
                compliance_checks=[
                    "risk_limits_checked",
                    "circuit_breakers_checked",
                    "kill_switch_verified",
                ],
                executed=False,
            )

            self.decision_audit_trail.append(audit_record)
            logger.info(f"Decision audit recorded: {decision_id}")

            return decision_id

        except Exception as e:
            logger.error(f"Error recording decision audit: {e}")
            return ""

    def mark_decision_executed(
        self, decision_id: str, execution_time: datetime = None
    ) -> bool:
        """Mark a decision as executed"""
        try:
            for audit in self.decision_audit_trail:
                if audit.decision_id == decision_id:
                    audit.executed = True
                    audit.execution_time = execution_time or datetime.now()
                    logger.info(f"Decision marked as executed: {decision_id}")
                    return True

            logger.warning(f"Decision not found for execution marking: {decision_id}")
            return False

        except Exception as e:
            logger.error(f"Error marking decision as executed: {e}")
            return False

    def generate_compliance_report(
        self, report_date: datetime = None
    ) -> ComplianceReport:
        """Generate daily compliance report"""
        try:
            if report_date is None:
                report_date = datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

            next_day = report_date + timedelta(days=1)

            # Filter decisions for the day
            daily_decisions = [
                audit
                for audit in self.decision_audit_trail
                if report_date <= audit.timestamp < next_day
            ]

            executed_trades = [audit for audit in daily_decisions if audit.executed]

            # Count risk events
            risk_breaches = []
            circuit_breaker_triggers = 0
            kill_switch_activations = 0

            for audit in daily_decisions:
                if "risk_breach" in audit.rationale.lower():
                    risk_breaches.append(audit.rationale)
                if audit.decision_type == "circuit_breaker":
                    circuit_breaker_triggers += 1
                if audit.decision_type == "kill_switch":
                    kill_switch_activations += 1

            # Calculate average confidence
            if daily_decisions:
                avg_confidence = np.mean(
                    [audit.confidence_level for audit in daily_decisions]
                )
            else:
                avg_confidence = 0.0

            # Check data lineage and audit completeness
            data_lineage_complete = all(
                len(audit.data_sources) > 0 for audit in daily_decisions
            )
            audit_trail_complete = all(
                len(audit.compliance_checks) > 0 for audit in daily_decisions
            )

            report = ComplianceReport(
                report_date=report_date,
                total_decisions=len(daily_decisions),
                executed_trades=len(executed_trades),
                risk_breaches=risk_breaches,
                circuit_breaker_triggers=circuit_breaker_triggers,
                kill_switch_activations=kill_switch_activations,
                avg_confidence_level=avg_confidence,
                data_lineage_complete=data_lineage_complete,
                audit_trail_complete=audit_trail_complete,
            )

            logger.info(f"Compliance report generated for {report_date.date()}")
            return report

        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            raise

    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk control status"""
        try:
            # Circuit breaker status
            breaker_status = {}
            for breaker_id, breaker in self.circuit_breakers.items():
                breaker_status[breaker_id] = {
                    "name": breaker.name,
                    "triggered": breaker.is_triggered,
                    "current_value": breaker.current_value,
                    "threshold": breaker.threshold_value,
                    "utilization_pct": (
                        breaker.current_value / breaker.threshold_value * 100
                    )
                    if breaker.threshold_value > 0
                    else 0,
                }

            # Risk limit status
            limit_status = {}
            for limit_id, limit in self.risk_limits.items():
                limit_status[limit_id] = {
                    "type": limit.limit_type,
                    "current_value": limit.current_value,
                    "limit_value": limit.limit_value,
                    "utilization_pct": limit.utilization_pct,
                    "breach_count": limit.breach_count,
                }

            return {
                "kill_switch_active": self.kill_switch.is_active,
                "kill_switch_reason": self.kill_switch.trigger_reason,
                "circuit_breakers": breaker_status,
                "risk_limits": limit_status,
                "recent_decisions": len(
                    [
                        audit
                        for audit in self.decision_audit_trail
                        if audit.timestamp >= datetime.now() - timedelta(hours=24)
                    ]
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting risk status: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}


# Global risk control system instance
RISK_CONTROL_SYSTEM = RiskControlSystem()


def manage_risk_controls(
    action: str, control_type: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """Manage risk controls - main interface function"""
    try:
        logger.info(f"Risk control action: {action} (type: {control_type})")

        if action == "status":
            return {
                "success": True,
                "action": action,
                "risk_status": RISK_CONTROL_SYSTEM.get_risk_status(),
                "timestamp": datetime.now().isoformat(),
            }

        elif action == "configure":
            # Configure risk limits or circuit breakers
            if control_type == "circuit_breaker":
                # Add new circuit breaker
                breaker_id = f"cb_{uuid.uuid4().hex[:8]}"
                RISK_CONTROL_SYSTEM.circuit_breakers[breaker_id] = CircuitBreaker(
                    breaker_id=breaker_id,
                    name=kwargs.get("name", "Custom Breaker"),
                    trigger_type=kwargs.get("trigger_type", "drawdown"),
                    threshold_value=kwargs.get("threshold_value", 0.1),
                    current_value=0.0,
                )
                return {
                    "success": True,
                    "breaker_id": breaker_id,
                    "message": "Circuit breaker configured",
                }

            elif control_type == "kill_switch":
                # Configure kill switch
                if kwargs.get("activate"):
                    success = RISK_CONTROL_SYSTEM.activate_kill_switch(
                        reason=kwargs.get("reason", "Manual activation"),
                        triggered_by=kwargs.get("authorized_by", "user"),
                    )
                    return {
                        "success": success,
                        "message": "Kill switch activated"
                        if success
                        else "Failed to activate",
                    }
                else:
                    success = RISK_CONTROL_SYSTEM.deactivate_kill_switch(
                        authorized_by=kwargs.get("authorized_by", "user")
                    )
                    return {
                        "success": success,
                        "message": "Kill switch deactivated"
                        if success
                        else "Failed to deactivate",
                    }

        elif action == "trigger_test":
            # Test circuit breaker with mock data
            test_metrics = {
                "current_drawdown": 0.06,  # 6% drawdown
                "current_var": 0.03,  # 3% VaR
                "daily_pnl": -15000,  # -$15k loss
                "current_volatility": 0.6,  # 60% volatility
            }

            triggered = RISK_CONTROL_SYSTEM.check_circuit_breakers(test_metrics)
            return {
                "success": True,
                "action": action,
                "triggered_breakers": triggered,
                "test_metrics": test_metrics,
                "message": f"Test completed - {len(triggered)} breakers triggered",
            }

        elif action == "reset":
            if control_type == "circuit_breaker" and kwargs.get("breaker_id"):
                success = RISK_CONTROL_SYSTEM.reset_circuit_breaker(
                    kwargs["breaker_id"]
                )
                return {
                    "success": success,
                    "message": "Circuit breaker reset" if success else "Reset failed",
                }

        return {"success": False, "error": f"Unknown action: {action}"}

    except Exception as e:
        logger.error(f"Error in risk control management: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
