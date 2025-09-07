#!/usr/bin/env python3
"""
Enhanced Trading Metrics

Implementation of advanced trading performance metrics based on agents-mike.md:
- Calmar Ratio
- Sortino Ratio
- Implementation Shortfall
- Latency tracking
- Risk-adjusted returns
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from models import (
    CalmarRatio,
    SortinoRatio,
    ImplementationShortfall,
    Trade,
)

logger = logging.getLogger(__name__)


class EnhancedMetricsEngine:
    """Engine for calculating advanced trading performance metrics"""

    def __init__(self):
        self.trades: List[Trade] = []
        self.execution_data: List[Dict[str, Any]] = []

    def add_trade(self, trade: Trade):
        """Add trade for metrics calculation"""
        self.trades.append(trade)

    def add_execution_data(self, execution_info: Dict[str, Any]):
        """Add execution data for implementation shortfall calculation"""
        self.execution_data.append(execution_info)

    def calculate_calmar_ratio(
        self, time_period: str = "1y", benchmark_return: Optional[float] = None
    ) -> CalmarRatio:
        """
        Calculate Calmar Ratio = Annual Return / Maximum Drawdown
        Higher is better, >1.0 is generally good
        """
        try:
            # Get returns for the period
            returns_data = self._get_period_returns(time_period)

            if len(returns_data) == 0:
                logger.warning("No return data available for Calmar ratio calculation")
                return CalmarRatio(
                    ratio=0.0,
                    annual_return=0.0,
                    max_drawdown=0.0,
                    period_years=1.0,
                    timestamp=datetime.now(),
                )

            # Calculate annual return
            total_return = np.prod([1 + r for r in returns_data]) - 1
            period_years = self._get_period_years(time_period)
            annual_return = (1 + total_return) ** (1 / period_years) - 1

            # Calculate maximum drawdown
            cumulative_returns = np.cumprod([1 + r for r in returns_data])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns))

            # Calculate Calmar ratio
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0

            logger.info(
                f"Calmar Ratio: {calmar_ratio:.3f} (return: {annual_return:.3%}, MDD: {max_drawdown:.3%})"
            )

            return CalmarRatio(
                ratio=calmar_ratio,
                annual_return=annual_return,
                max_drawdown=max_drawdown,
                period_years=period_years,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            raise

    def calculate_sortino_ratio(
        self,
        time_period: str = "1y",
        target_return: float = 0.0,
        risk_free_rate: float = 0.02,
    ) -> SortinoRatio:
        """
        Calculate Sortino Ratio = (Return - Target) / Downside Deviation
        Focuses on downside risk only, unlike Sharpe ratio
        """
        try:
            # Get returns for the period
            returns_data = self._get_period_returns(time_period)

            if len(returns_data) == 0:
                logger.warning("No return data available for Sortino ratio calculation")
                return SortinoRatio(
                    ratio=0.0,
                    annual_return=0.0,
                    downside_deviation=0.0,
                    target_return=target_return,
                    timestamp=datetime.now(),
                )

            # Calculate annualized return
            total_return = np.prod([1 + r for r in returns_data]) - 1
            period_years = self._get_period_years(time_period)
            annual_return = (1 + total_return) ** (1 / period_years) - 1

            # Calculate downside deviation (only negative deviations from target)
            daily_target = target_return / 252  # Assuming 252 trading days
            downside_returns = [
                r - daily_target for r in returns_data if r < daily_target
            ]

            if len(downside_returns) == 0:
                downside_deviation = 0.0
            else:
                downside_variance = np.var(downside_returns, ddof=1)
                downside_deviation = np.sqrt(downside_variance * 252)  # Annualized

            # Calculate Sortino ratio
            sortino_ratio = (
                (annual_return - target_return) / downside_deviation
                if downside_deviation > 0
                else 0.0
            )

            logger.info(
                f"Sortino Ratio: {sortino_ratio:.3f} (return: {annual_return:.3%}, DD: {downside_deviation:.3%})"
            )

            return SortinoRatio(
                ratio=sortino_ratio,
                annual_return=annual_return,
                downside_deviation=downside_deviation,
                target_return=target_return,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            raise

    def calculate_implementation_shortfall(
        self, execution_id: str = None, time_period: str = "1d"
    ) -> ImplementationShortfall:
        """
        Calculate Implementation Shortfall = (Executed Price - Decision Price) / Decision Price
        Measures the cost of execution versus ideal execution
        """
        try:
            # Filter execution data
            if execution_id:
                relevant_executions = [
                    ex
                    for ex in self.execution_data
                    if ex.get("execution_id") == execution_id
                ]
            else:
                cutoff_date = datetime.now() - self._parse_time_period(time_period)
                relevant_executions = [
                    ex
                    for ex in self.execution_data
                    if ex.get("timestamp", datetime.min) >= cutoff_date
                ]

            if len(relevant_executions) == 0:
                logger.warning(
                    "No execution data available for implementation shortfall"
                )
                return ImplementationShortfall(
                    shortfall_bps=0.0,
                    market_impact_bps=0.0,
                    timing_cost_bps=0.0,
                    opportunity_cost_bps=0.0,
                    trade_value=0.0,
                    execution_time_seconds=0.0,
                    timestamp=datetime.now(),
                )

            total_shortfall_bps = 0.0
            total_market_impact_bps = 0.0
            total_timing_cost_bps = 0.0
            total_opportunity_cost_bps = 0.0
            total_trade_value = 0.0
            total_execution_time = 0.0

            for execution in relevant_executions:
                decision_price = execution.get("decision_price", 0.0)
                executed_price = execution.get("executed_price", 0.0)
                trade_value = execution.get("trade_value", 0.0)
                execution_time = execution.get("execution_time_seconds", 0.0)

                # Calculate shortfall in basis points
                if decision_price > 0:
                    shortfall = (executed_price - decision_price) / decision_price
                    shortfall_bps = shortfall * 10000  # Convert to bps

                    # Decompose into components (simplified)
                    market_impact = execution.get(
                        "market_impact_bps", shortfall_bps * 0.6
                    )
                    timing_cost = execution.get("timing_cost_bps", shortfall_bps * 0.3)
                    opportunity_cost = execution.get(
                        "opportunity_cost_bps", shortfall_bps * 0.1
                    )

                    # Weight by trade value
                    weight = trade_value
                    total_shortfall_bps += shortfall_bps * weight
                    total_market_impact_bps += market_impact * weight
                    total_timing_cost_bps += timing_cost * weight
                    total_opportunity_cost_bps += opportunity_cost * weight
                    total_trade_value += weight
                    total_execution_time += execution_time

            # Calculate weighted averages
            if total_trade_value > 0:
                avg_shortfall_bps = total_shortfall_bps / total_trade_value
                avg_market_impact_bps = total_market_impact_bps / total_trade_value
                avg_timing_cost_bps = total_timing_cost_bps / total_trade_value
                avg_opportunity_cost_bps = (
                    total_opportunity_cost_bps / total_trade_value
                )
            else:
                avg_shortfall_bps = avg_market_impact_bps = avg_timing_cost_bps = (
                    avg_opportunity_cost_bps
                ) = 0.0

            avg_execution_time = total_execution_time / len(relevant_executions)

            logger.info(
                f"Implementation Shortfall: {avg_shortfall_bps:.2f} bps (market: {avg_market_impact_bps:.2f}, timing: {avg_timing_cost_bps:.2f})"
            )

            return ImplementationShortfall(
                shortfall_bps=avg_shortfall_bps,
                market_impact_bps=avg_market_impact_bps,
                timing_cost_bps=avg_timing_cost_bps,
                opportunity_cost_bps=avg_opportunity_cost_bps,
                trade_value=total_trade_value,
                execution_time_seconds=avg_execution_time,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error calculating implementation shortfall: {e}")
            raise

    def calculate_comprehensive_metrics(
        self, time_period: str = "1y", benchmark: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate all enhanced metrics in one go"""
        try:
            logger.info(f"Calculating comprehensive metrics for period: {time_period}")

            # Calculate all metrics
            calmar = self.calculate_calmar_ratio(time_period)
            sortino = self.calculate_sortino_ratio(time_period)
            impl_shortfall = self.calculate_implementation_shortfall(
                time_period=time_period
            )

            # Additional metrics
            returns_data = self._get_period_returns(time_period)
            additional_metrics = self._calculate_additional_metrics(
                returns_data, benchmark
            )

            return {
                "time_period": time_period,
                "calmar_ratio": calmar.model_dump(),
                "sortino_ratio": sortino.model_dump(),
                "implementation_shortfall": impl_shortfall.model_dump(),
                "additional_metrics": additional_metrics,
                "benchmark": benchmark,
                "calculation_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _get_period_returns(self, time_period: str) -> List[float]:
        """Get returns for specified time period"""
        try:
            cutoff_date = datetime.now() - self._parse_time_period(time_period)

            # Filter trades by date
            period_trades = [
                trade for trade in self.trades if trade.timestamp >= cutoff_date
            ]

            if len(period_trades) == 0:
                return []

            # Calculate daily returns (simplified)
            # In practice, you'd need proper portfolio valuation
            returns = []
            for trade in period_trades:
                if hasattr(trade, "return_pct"):
                    returns.append(trade.return_pct)
                else:
                    # Estimate return from trade data
                    if trade.quantity != 0:
                        return_est = (trade.price - trade.avg_cost) / trade.avg_cost
                        returns.append(return_est)

            return returns

        except Exception as e:
            logger.error(f"Error getting period returns: {e}")
            return []

    def _parse_time_period(self, time_period: str) -> timedelta:
        """Parse time period string to timedelta"""
        if time_period.endswith("d"):
            days = int(time_period[:-1])
            return timedelta(days=days)
        elif time_period.endswith("w"):
            weeks = int(time_period[:-1])
            return timedelta(weeks=weeks)
        elif time_period.endswith("m"):
            months = int(time_period[:-1])
            return timedelta(days=months * 30)
        elif time_period.endswith("y"):
            years = int(time_period[:-1])
            return timedelta(days=years * 365)
        else:
            return timedelta(days=365)  # Default to 1 year

    def _get_period_years(self, time_period: str) -> float:
        """Get period length in years"""
        td = self._parse_time_period(time_period)
        return td.days / 365.25

    def _calculate_additional_metrics(
        self, returns_data: List[float], benchmark: Optional[str]
    ) -> Dict[str, Any]:
        """Calculate additional performance metrics"""
        try:
            if len(returns_data) == 0:
                return {
                    "sharpe_ratio": 0.0,
                    "information_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "volatility": 0.0,
                }

            # Convert to numpy array
            returns = np.array(returns_data)

            # Sharpe ratio (using 2% risk-free rate)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            sharpe_ratio = (
                np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) > 0
                else 0.0
            )

            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns))

            # Win rate
            positive_returns = returns[returns > 0]
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

            # Profit factor
            gross_profit = (
                np.sum(positive_returns) if len(positive_returns) > 0 else 0.0
            )
            gross_loss = (
                abs(np.sum(returns[returns < 0]))
                if len(returns[returns < 0]) > 0
                else 0.0
            )
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

            # Volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)

            # Information ratio (vs benchmark - simplified)
            information_ratio = (
                sharpe_ratio  # Simplified - would need actual benchmark data
            )

            return {
                "sharpe_ratio": float(sharpe_ratio),
                "information_ratio": float(information_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "profit_factor": float(profit_factor),
                "volatility": float(volatility),
                "total_trades": len(returns),
                "positive_trades": len(positive_returns),
                "negative_trades": len(returns) - len(positive_returns),
            }

        except Exception as e:
            logger.error(f"Error calculating additional metrics: {e}")
            return {"error": str(e)}


# Global enhanced metrics engine
ENHANCED_METRICS_ENGINE = EnhancedMetricsEngine()


def calculate_enhanced_metrics(
    metrics: List[str], time_period: str = "1y", benchmark: Optional[str] = None
) -> Dict[str, Any]:
    """Main interface for calculating enhanced trading metrics"""
    try:
        logger.info(f"Calculating enhanced metrics: {metrics} for period {time_period}")

        results = {
            "requested_metrics": metrics,
            "time_period": time_period,
            "benchmark": benchmark,
            "calculated_metrics": {},
            "success": True,
            "timestamp": datetime.now().isoformat(),
        }

        # Calculate requested metrics
        if "calmar" in metrics:
            calmar = ENHANCED_METRICS_ENGINE.calculate_calmar_ratio(time_period)
            results["calculated_metrics"]["calmar_ratio"] = calmar.model_dump()

        if "sortino" in metrics:
            sortino = ENHANCED_METRICS_ENGINE.calculate_sortino_ratio(time_period)
            results["calculated_metrics"]["sortino_ratio"] = sortino.model_dump()

        if "implementation_shortfall" in metrics:
            impl_shortfall = ENHANCED_METRICS_ENGINE.calculate_implementation_shortfall(
                time_period=time_period
            )
            results["calculated_metrics"]["implementation_shortfall"] = (
                impl_shortfall.model_dump()
            )

        # Add comprehensive metrics if all requested
        if set(metrics) == {"calmar", "sortino", "implementation_shortfall"}:
            comprehensive = ENHANCED_METRICS_ENGINE.calculate_comprehensive_metrics(
                time_period, benchmark
            )
            results["comprehensive_analysis"] = comprehensive

        # Generate recommendations
        results["recommendations"] = _generate_performance_recommendations(
            results["calculated_metrics"]
        )

        return results

    except Exception as e:
        logger.error(f"Error calculating enhanced metrics: {e}")
        return {
            "success": False,
            "error": str(e),
            "requested_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }


def _generate_performance_recommendations(metrics: Dict[str, Any]) -> List[str]:
    """Generate performance recommendations based on calculated metrics"""
    recommendations = []

    # Calmar ratio recommendations
    if "calmar_ratio" in metrics:
        calmar = metrics["calmar_ratio"]["ratio"]
        if calmar > 2.0:
            recommendations.append(
                "🏆 Excellent Calmar ratio - strong risk-adjusted returns"
            )
        elif calmar > 1.0:
            recommendations.append(
                "✅ Good Calmar ratio - acceptable risk-adjusted performance"
            )
        elif calmar > 0.5:
            recommendations.append(
                "⚠️ Moderate Calmar ratio - consider reducing risk or improving returns"
            )
        else:
            recommendations.append(
                "❌ Poor Calmar ratio - significant improvements needed"
            )

    # Sortino ratio recommendations
    if "sortino_ratio" in metrics:
        sortino = metrics["sortino_ratio"]["ratio"]
        if sortino > 2.0:
            recommendations.append(
                "📈 Excellent Sortino ratio - strong downside risk management"
            )
        elif sortino > 1.0:
            recommendations.append(
                "✅ Good Sortino ratio - adequate downside protection"
            )
        else:
            recommendations.append(
                "📉 Low Sortino ratio - focus on reducing downside risk"
            )

    # Implementation shortfall recommendations
    if "implementation_shortfall" in metrics:
        shortfall = metrics["implementation_shortfall"]["shortfall_bps"]
        if abs(shortfall) < 10:
            recommendations.append(
                "🎯 Excellent execution - minimal implementation shortfall"
            )
        elif abs(shortfall) < 25:
            recommendations.append(
                "✅ Good execution quality - acceptable implementation costs"
            )
        elif abs(shortfall) < 50:
            recommendations.append(
                "⚠️ Moderate execution costs - consider optimizing order timing"
            )
        else:
            recommendations.append(
                "❌ High implementation shortfall - review execution strategy"
            )

    return recommendations


def add_trade_for_metrics(trade: Trade):
    """Helper function to add trade data for metrics calculation"""
    ENHANCED_METRICS_ENGINE.add_trade(trade)


def add_execution_data(execution_info: Dict[str, Any]):
    """Helper function to add execution data for implementation shortfall"""
    ENHANCED_METRICS_ENGINE.add_execution_data(execution_info)
