#!/usr/bin/env python3
"""
Forecast Calibration Engine

Implementation of forecast calibration metrics based on agents-mike.md recommendations:
- Brier Score calculation
- Expected Calibration Error (ECE)
- Reliability diagrams
- Calibration monitoring
"""

import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

from models import (
    ForecastCalibration,
    BrierScore,
    CalibrationBin,
    ExpectedCalibrationError,
    ReliabilityDiagram,
    ForecastResult,
)

logger = logging.getLogger(__name__)


class CalibrationEngine:
    """Engine for forecast calibration analysis and monitoring"""

    def __init__(self):
        self.forecasts: List[ForecastCalibration] = []
        self.resolved_forecasts: List[ForecastCalibration] = []

    def register_forecast(self, forecast_result: ForecastResult, question: str) -> str:
        """Register a new forecast for calibration tracking"""
        import uuid

        forecast_id = f"forecast_{uuid.uuid4().hex[:8]}"

        calibration_forecast = ForecastCalibration(
            forecast_id=forecast_id,
            predicted_probability=forecast_result.prediction_probability,
            actual_outcome=None,  # To be resolved later
            question=question,
            forecast_date=forecast_result.timestamp,
            resolution_date=None,
            confidence_interval=forecast_result.confidence_interval,
            agent_source=forecast_result.agent_type,
        )

        self.forecasts.append(calibration_forecast)
        logger.info(
            f"Registered forecast {forecast_id}: {forecast_result.prediction_probability:.3f} probability"
        )

        return forecast_id

    def resolve_forecast(self, forecast_id: str, actual_outcome: bool) -> bool:
        """Resolve a forecast with actual outcome"""
        for forecast in self.forecasts:
            if forecast.forecast_id == forecast_id:
                forecast.actual_outcome = actual_outcome
                forecast.resolution_date = datetime.now()
                self.resolved_forecasts.append(forecast)

                logger.info(
                    f"Resolved forecast {forecast_id}: predicted {forecast.predicted_probability:.3f}, actual {actual_outcome}"
                )
                return True

        logger.warning(f"Forecast {forecast_id} not found for resolution")
        return False

    def calculate_brier_score(self, time_period: str = "30d") -> BrierScore:
        """
        Calculate Brier Score for resolved forecasts

        Brier Score = (1/N) * Σ(forecast_prob - actual_outcome)²
        Lower is better, perfect score = 0
        """
        try:
            # Filter forecasts by time period
            cutoff_date = self._get_cutoff_date(time_period)
            recent_forecasts = [
                f
                for f in self.resolved_forecasts
                if f.forecast_date >= cutoff_date and f.actual_outcome is not None
            ]

            if len(recent_forecasts) == 0:
                logger.warning(
                    "No resolved forecasts found for Brier score calculation"
                )
                return BrierScore(
                    score=1.0,  # Worst possible score
                    num_forecasts=0,
                    resolution_period=time_period,
                    skill_score=None,
                    timestamp=datetime.now(),
                )

            # Calculate Brier score
            brier_scores = []
            for forecast in recent_forecasts:
                prob = forecast.predicted_probability
                outcome = float(forecast.actual_outcome)
                brier_scores.append((prob - outcome) ** 2)

            brier_score = np.mean(brier_scores)

            # Calculate skill score vs random baseline (0.25 for binary)
            baseline_score = 0.25  # Expected Brier score for random predictions
            skill_score = (
                1 - (brier_score / baseline_score) if baseline_score > 0 else None
            )

            skill_str = f"{skill_score:.4f}" if skill_score is not None else "N/A"
            logger.info(f"Brier Score: {brier_score:.4f} (skill: {skill_str})")

            return BrierScore(
                score=brier_score,
                num_forecasts=len(recent_forecasts),
                resolution_period=time_period,
                skill_score=skill_score,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error calculating Brier score: {e}")
            raise

    def calculate_expected_calibration_error(
        self, time_period: str = "30d", num_bins: int = 10
    ) -> ExpectedCalibrationError:
        """
        Calculate Expected Calibration Error (ECE)

        ECE = Σ (|bin_accuracy - bin_confidence|) * bin_weight
        """
        try:
            # Get recent resolved forecasts
            cutoff_date = self._get_cutoff_date(time_period)
            recent_forecasts = [
                f
                for f in self.resolved_forecasts
                if f.forecast_date >= cutoff_date and f.actual_outcome is not None
            ]

            if len(recent_forecasts) == 0:
                logger.warning("No resolved forecasts found for ECE calculation")
                return ExpectedCalibrationError(
                    ece=1.0,
                    max_calibration_error=1.0,
                    bins=[],
                    num_bins=num_bins,
                    timestamp=datetime.now(),
                )

            # Create bins
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bins = []

            total_forecasts = len(recent_forecasts)
            ece_sum = 0.0
            max_calibration_error = 0.0

            for i in range(num_bins):
                bin_lower = bin_edges[i]
                bin_upper = bin_edges[i + 1]
                bin_center = (bin_lower + bin_upper) / 2

                # Find forecasts in this bin
                bin_forecasts = [
                    f
                    for f in recent_forecasts
                    if bin_lower <= f.predicted_probability < bin_upper
                ]

                if len(bin_forecasts) == 0:
                    continue

                # Calculate bin accuracy and confidence
                bin_predictions = [f.predicted_probability for f in bin_forecasts]
                bin_outcomes = [float(f.actual_outcome) for f in bin_forecasts]

                avg_confidence = np.mean(bin_predictions)
                avg_accuracy = np.mean(bin_outcomes)
                bin_weight = len(bin_forecasts) / total_forecasts

                # Contribution to ECE
                calibration_error = abs(avg_accuracy - avg_confidence)
                ece_sum += calibration_error * bin_weight
                max_calibration_error = max(max_calibration_error, calibration_error)

                bins.append(
                    CalibrationBin(
                        bin_center=bin_center,
                        predicted_prob=avg_confidence,
                        actual_freq=avg_accuracy,
                        count=len(bin_forecasts),
                    )
                )

            logger.info(f"ECE: {ece_sum:.4f}, MCE: {max_calibration_error:.4f}")

            return ExpectedCalibrationError(
                ece=ece_sum,
                max_calibration_error=max_calibration_error,
                bins=bins,
                num_bins=num_bins,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error calculating ECE: {e}")
            raise

    def generate_reliability_diagram(
        self, time_period: str = "30d"
    ) -> ReliabilityDiagram:
        """Generate reliability diagram data for calibration visualization"""
        try:
            # Calculate ECE first to get bins
            ece_result = self.calculate_expected_calibration_error(time_period)
            brier_result = self.calculate_brier_score(time_period)

            # Perfect calibration line (diagonal)
            perfect_line = [i / 10 for i in range(11)]  # [0.0, 0.1, ..., 1.0]

            return ReliabilityDiagram(
                bins=ece_result.bins,
                perfect_calibration_line=perfect_line,
                ece_score=ece_result.ece,
                brier_score=brier_result.score,
                num_forecasts=brier_result.num_forecasts,
            )

        except Exception as e:
            logger.error(f"Error generating reliability diagram: {e}")
            raise

    def get_calibration_summary(self, time_period: str = "30d") -> Dict[str, Any]:
        """Get comprehensive calibration summary"""
        try:
            cutoff_date = self._get_cutoff_date(time_period)

            # Get forecast counts
            total_forecasts = len(self.forecasts)
            resolved_forecasts = len(
                [f for f in self.forecasts if f.actual_outcome is not None]
            )
            recent_resolved = len(
                [f for f in self.resolved_forecasts if f.forecast_date >= cutoff_date]
            )

            # Calculate metrics
            brier_score = self.calculate_brier_score(time_period)
            ece_result = self.calculate_expected_calibration_error(time_period)

            # Agent performance breakdown
            agent_performance = {}
            if len(self.resolved_forecasts) > 0:
                for agent_type in set(f.agent_source for f in self.resolved_forecasts):
                    agent_forecasts = [
                        f
                        for f in self.resolved_forecasts
                        if f.agent_source == agent_type
                        and f.forecast_date >= cutoff_date
                    ]

                    if len(agent_forecasts) > 0:
                        try:
                            agent_brier = np.mean(
                                [
                                    (f.predicted_probability - float(f.actual_outcome))
                                    ** 2
                                    for f in agent_forecasts
                                    if f.actual_outcome is not None
                                ]
                            )
                            agent_performance[agent_type] = {
                                "count": len(agent_forecasts),
                                "brier_score": agent_brier,
                            }
                        except Exception as e:
                            logger.warning(
                                f"Error calculating agent performance for {agent_type}: {e}"
                            )

            return {
                "time_period": time_period,
                "forecast_counts": {
                    "total_registered": total_forecasts,
                    "total_resolved": resolved_forecasts,
                    "recent_resolved": recent_resolved,
                },
                "calibration_metrics": {
                    "brier_score": brier_score.score if brier_score else None,
                    "brier_skill_score": brier_score.skill_score
                    if brier_score
                    else None,
                    "expected_calibration_error": ece_result.ece
                    if ece_result
                    else None,
                    "max_calibration_error": ece_result.max_calibration_error
                    if ece_result
                    else None,
                },
                "agent_performance": agent_performance,
                "calibration_quality": self._assess_calibration_quality(
                    brier_score.score
                    if brier_score and hasattr(brier_score, "score")
                    else None,
                    ece_result.ece
                    if ece_result and hasattr(ece_result, "ece")
                    else None,
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generating calibration summary: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _get_cutoff_date(self, time_period: str) -> datetime:
        """Parse time period and return cutoff date"""
        now = datetime.now()

        if time_period.endswith("d"):
            days = int(time_period[:-1])
            return now - timedelta(days=days)
        elif time_period.endswith("w"):
            weeks = int(time_period[:-1])
            return now - timedelta(weeks=weeks)
        elif time_period.endswith("m"):
            months = int(time_period[:-1])
            return now - timedelta(days=months * 30)  # Approximate
        else:
            return now - timedelta(days=30)  # Default to 30 days

    def _assess_calibration_quality(self, brier_score: float, ece: float) -> str:
        """Assess overall calibration quality"""
        # Handle None values
        if brier_score is None or ece is None:
            return "No Data"
        if brier_score <= 0.1 and ece <= 0.05:
            return "Excellent"
        elif brier_score <= 0.2 and ece <= 0.1:
            return "Good"
        elif brier_score <= 0.3 and ece <= 0.15:
            return "Acceptable"
        else:
            return "Poor"


# Global calibration engine instance
CALIBRATION_ENGINE = CalibrationEngine()


def analyze_forecast_calibration(
    time_period: str = "30d",
    agent_types: List[str] = None,
    include_reliability_diagram: bool = True,
) -> Dict[str, Any]:
    """Analyze forecast calibration for specified period and agents"""
    try:
        logger.info(f"Analyzing forecast calibration for period: {time_period}")

        # Get calibration summary
        summary = CALIBRATION_ENGINE.get_calibration_summary(time_period)

        # Add reliability diagram if requested
        if (
            include_reliability_diagram
            and summary.get("forecast_counts", {}).get("recent_resolved", 0) > 0
        ):
            try:
                reliability_diagram = CALIBRATION_ENGINE.generate_reliability_diagram(
                    time_period
                )
                summary["reliability_diagram"] = reliability_diagram.model_dump()
            except Exception as e:
                logger.warning(f"Could not generate reliability diagram: {e}")
                summary["reliability_diagram"] = None

        # Filter by agent types if specified
        if agent_types and "all" not in agent_types:
            filtered_performance = {
                agent: perf
                for agent, perf in summary.get("agent_performance", {}).items()
                if agent in agent_types
            }
            summary["agent_performance"] = filtered_performance

        return {
            "success": True,
            "calibration_analysis": summary,
            "time_period": time_period,
            "agent_types_analyzed": agent_types or ["all"],
            "recommendations": _generate_calibration_recommendations(summary),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in calibration analysis: {e}")
        return {
            "success": False,
            "error": str(e),
            "time_period": time_period,
            "timestamp": datetime.now().isoformat(),
        }


def _generate_calibration_recommendations(summary: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on calibration metrics"""
    recommendations = []

    metrics = summary.get("calibration_metrics", {})
    brier_score = metrics.get("brier_score")
    ece = metrics.get("expected_calibration_error")
    quality = summary.get("calibration_quality", "Poor")

    if quality == "Poor":
        recommendations.append(
            "⚠️ Poor calibration detected - consider retraining forecast models"
        )

    if brier_score is not None and brier_score > 0.25:
        recommendations.append(
            "📉 High Brier score - forecasts are worse than random baseline"
        )

    if ece is not None and ece > 0.1:
        recommendations.append(
            "📊 High calibration error - predicted probabilities don't match actual frequencies"
        )

    skill_score = metrics.get("brier_skill_score")
    if skill_score is not None and skill_score < 0:
        recommendations.append(
            "🎲 Negative skill score - random predictions would perform better"
        )

    # Agent-specific recommendations
    agent_performance = summary.get("agent_performance", {})
    if len(agent_performance) > 1:
        best_agent = min(agent_performance.items(), key=lambda x: x[1]["brier_score"])
        worst_agent = max(agent_performance.items(), key=lambda x: x[1]["brier_score"])

        recommendations.append(
            f"🏆 Best performing agent: {best_agent[0]} (Brier: {best_agent[1]['brier_score']:.3f})"
        )
        recommendations.append(
            f"📉 Worst performing agent: {worst_agent[0]} (Brier: {worst_agent[1]['brier_score']:.3f})"
        )

    if quality in ["Good", "Excellent"]:
        recommendations.append(
            "✅ Good calibration - forecast probabilities are reliable"
        )

    return recommendations


def register_forecast_for_calibration(
    forecast_result: ForecastResult, question: str
) -> str:
    """Helper function to register forecast for calibration tracking"""
    return CALIBRATION_ENGINE.register_forecast(forecast_result, question)


def resolve_forecast_outcome(forecast_id: str, actual_outcome: bool) -> bool:
    """Helper function to resolve forecast outcome"""
    return CALIBRATION_ENGINE.resolve_forecast(forecast_id, actual_outcome)
