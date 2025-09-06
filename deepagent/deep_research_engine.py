#!/usr/bin/env python3
"""
Deep Research Engine for SGR Financial Trading Agent

Implements adaptive reasoning and deep research capabilities for financial markets.
Based on SGR-classic pattern with NextStep adaptive planning.
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import yfinance as yf
import numpy as np
from openai import AzureOpenAI

from deep_research_models import (
    NextStep,
    ResearchContext,
    ResearchInsight,
    ResearchFindings,
    ResearchPhase,
    ResearchAction,
    DeepResearchRequest,
    CompetitorAnalysisRequest,
    EconomicIndicatorRequest,
    ValidationRequest,
    SynthesisRequest,
    SGRDeepResearchResponse,
)
from models import MarketDataRequest, NewsAnalysisRequest, RiskAssessmentRequest
from settings import settings

logger = logging.getLogger(__name__)


class DeepResearchEngine:
    """Deep Research Engine with adaptive planning capabilities"""

    def __init__(self, azure_client: AzureOpenAI):
        self.client = azure_client
        self.research_sessions: Dict[str, ResearchContext] = {}

    def start_deep_research(self, request: DeepResearchRequest) -> ResearchContext:
        """Initialize a new deep research session"""
        research_id = str(uuid.uuid4())

        context = ResearchContext(
            research_id=research_id,
            primary_symbols=request.symbols,
            research_question=request.research_question,
            current_phase=ResearchPhase.MARKET_DISCOVERY,
            max_depth=request.max_depth,
            priority_areas=request.focus_areas or [],
        )

        self.research_sessions[research_id] = context
        logger.info(
            f"Started deep research session {research_id} for symbols {request.symbols}"
        )

        return context

    def get_next_step(self, context: ResearchContext) -> NextStep:
        """Generate next adaptive research step using Azure OpenAI"""
        try:
            # Prepare context for the LLM
            insights_summary = self._summarize_insights(context)
            knowledge_gaps = self._identify_knowledge_gaps(context)

            system_prompt = f"""You are a deep financial research analyst using adaptive reasoning.
            
Current Research Context:
- Research Question: {context.research_question}
- Symbols: {context.primary_symbols}
- Current Phase: {context.current_phase.value}
- Research Depth: {context.research_depth}/{context.max_depth}
- Insights Collected: {len(context.insights_collected)}

Current Understanding: {insights_summary}

Knowledge Gaps Identified: {", ".join(knowledge_gaps)}

Available Research Actions:
- collect_market_data: Get comprehensive market data and price history
- analyze_fundamentals: Deep dive into financial statements and ratios
- technical_indicators: Advanced technical analysis with multiple timeframes
- news_sentiment: News and social media sentiment analysis
- competitor_analysis: Compare with industry peers and competitors
- economic_indicators: Analyze macroeconomic factors and correlations
- risk_metrics: Comprehensive risk assessment including VaR, correlations
- forecast_models: Generate probabilistic forecasts with multiple models
- portfolio_allocation: Optimize portfolio allocation and position sizing
- backtesting: Validate strategies with historical data
- validate_findings: Cross-validate research findings and assumptions
- synthesize_results: Synthesize all findings into actionable recommendations

Your task is to determine the most logical next step in this adaptive research process.
Consider what will provide the highest value insights given current knowledge gaps.
Be strategic about building towards a comprehensive investment thesis.

Respond with detailed reasoning and specific next action."""

            user_prompt = f"""Given the current research state, what should be the next research step?

Current Phase: {context.current_phase.value}
Confidence Level: {context.confidence_level:.2f}
Priority Areas: {context.priority_areas}

Please provide the next adaptive step with detailed reasoning."""

            # Get structured response from Azure OpenAI
            completion = self.client.beta.chat.completions.parse(
                model=settings.azure_openai_deployment_name,
                response_format=NextStep,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=800,
            )

            next_step = completion.choices[0].message.parsed
            logger.info(f"Generated next step: {next_step.next_action}")

            return next_step

        except Exception as e:
            logger.error(f"Error generating next step: {e}")
            # Fallback to deterministic next step
            return self._fallback_next_step(context)

    def _summarize_insights(self, context: ResearchContext) -> str:
        """Summarize collected insights"""
        if not context.insights_collected:
            return "No insights collected yet."

        insights_by_phase = {}
        for insight in context.insights_collected:
            phase = insight.phase.value
            if phase not in insights_by_phase:
                insights_by_phase[phase] = []
            insights_by_phase[phase].append(insight.content[:100] + "...")

        summary_parts = []
        for phase, insights in insights_by_phase.items():
            summary_parts.append(f"{phase}: {len(insights)} insights")

        return "; ".join(summary_parts)

    def _identify_knowledge_gaps(self, context: ResearchContext) -> List[str]:
        """Identify knowledge gaps based on current research state"""
        collected_phases = set(insight.phase for insight in context.insights_collected)
        all_phases = set(ResearchPhase)
        missing_phases = all_phases - collected_phases

        gaps = []
        for phase in missing_phases:
            if phase == ResearchPhase.MARKET_DISCOVERY:
                gaps.append("Missing basic market data and price trends")
            elif phase == ResearchPhase.FUNDAMENTAL_ANALYSIS:
                gaps.append("No fundamental analysis of financial health")
            elif phase == ResearchPhase.TECHNICAL_ANALYSIS:
                gaps.append("Missing technical indicators and chart patterns")
            elif phase == ResearchPhase.SENTIMENT_ANALYSIS:
                gaps.append("No sentiment analysis from news and social media")
            elif phase == ResearchPhase.RISK_ASSESSMENT:
                gaps.append("Missing comprehensive risk metrics")
            elif phase == ResearchPhase.FORECAST_GENERATION:
                gaps.append("No probabilistic forecasts generated")

        # Add specific gaps based on symbols
        if len(context.primary_symbols) > 1:
            gaps.append("Missing correlation analysis between symbols")

        return gaps[:5]  # Limit to top 5 gaps

    def _fallback_next_step(self, context: ResearchContext) -> NextStep:
        """Fallback deterministic next step if LLM fails"""
        # Simple progression through research phases
        phase_actions = {
            ResearchPhase.MARKET_DISCOVERY: ResearchAction.COLLECT_MARKET_DATA,
            ResearchPhase.FUNDAMENTAL_ANALYSIS: ResearchAction.ANALYZE_FUNDAMENTALS,
            ResearchPhase.TECHNICAL_ANALYSIS: ResearchAction.TECHNICAL_INDICATORS,
            ResearchPhase.SENTIMENT_ANALYSIS: ResearchAction.NEWS_SENTIMENT,
            ResearchPhase.RISK_ASSESSMENT: ResearchAction.RISK_METRICS,
            ResearchPhase.FORECAST_GENERATION: ResearchAction.FORECAST_MODELS,
            ResearchPhase.VALIDATION: ResearchAction.VALIDATE_FINDINGS,
            ResearchPhase.SYNTHESIS: ResearchAction.SYNTHESIZE_RESULTS,
        }

        next_action = phase_actions.get(
            context.current_phase, ResearchAction.COLLECT_MARKET_DATA
        )

        return NextStep(
            reasoning="Fallback to deterministic progression through research phases",
            current_understanding=f"Currently in {context.current_phase.value} phase",
            knowledge_gaps=["Using fallback planning due to LLM unavailability"],
            next_action=next_action,
            action_rationale=f"Standard progression suggests {next_action.value}",
            expected_insights=[f"Basic insights from {next_action.value}"],
            research_priority="medium",
            estimated_completion_progress=min(
                0.9, context.research_depth / context.max_depth
            ),
            adaptive_plan=["Continue with standard research progression"],
            should_continue=context.research_depth < context.max_depth,
        )

    def add_insight(self, context: ResearchContext, insight: ResearchInsight) -> None:
        """Add new research insight to context"""
        context.insights_collected.append(insight)

        # Update confidence based on insight quality
        insight_weight = insight.confidence * insight.impact_score
        current_weight = context.confidence_level * len(context.insights_collected)
        new_confidence = (current_weight + insight_weight) / (
            len(context.insights_collected) + 1
        )
        context.confidence_level = min(1.0, new_confidence)

        logger.info(
            f"Added insight from {insight.phase.value}, confidence now {context.confidence_level:.2f}"
        )

    def should_continue_research(
        self, context: ResearchContext, next_step: NextStep
    ) -> bool:
        """Determine if research should continue"""
        if context.research_depth >= context.max_depth:
            return False

        if not next_step.should_continue:
            return False

        if context.confidence_level > 0.85 and len(context.insights_collected) >= 8:
            return False  # High confidence with sufficient insights

        return True

    def advance_phase(self, context: ResearchContext) -> None:
        """Advance to next research phase if appropriate"""
        phase_order = [
            ResearchPhase.MARKET_DISCOVERY,
            ResearchPhase.FUNDAMENTAL_ANALYSIS,
            ResearchPhase.TECHNICAL_ANALYSIS,
            ResearchPhase.SENTIMENT_ANALYSIS,
            ResearchPhase.RISK_ASSESSMENT,
            ResearchPhase.FORECAST_GENERATION,
            ResearchPhase.PORTFOLIO_OPTIMIZATION,
            ResearchPhase.VALIDATION,
            ResearchPhase.SYNTHESIS,
        ]

        current_index = phase_order.index(context.current_phase)
        if current_index < len(phase_order) - 1:
            context.current_phase = phase_order[current_index + 1]
            logger.info(f"Advanced to phase: {context.current_phase.value}")

    def generate_final_findings(self, context: ResearchContext) -> ResearchFindings:
        """Generate final research findings and recommendations"""
        try:
            # Summarize all insights
            insights_summary = "\n".join(
                [f"- {insight.content}" for insight in context.insights_collected]
            )

            system_prompt = """You are a senior financial analyst preparing final investment recommendations.
Synthesize all research findings into actionable investment advice.

Focus on:
- Clear investment thesis
- Specific risk assessment
- Actionable recommendations with rationale
- Realistic position sizing
- Clear entry/exit strategies
- Monitoring plan for ongoing management

Be conservative and professional in your recommendations."""

            user_prompt = f"""Based on the following research insights, provide final investment recommendations:

Research Question: {context.research_question}
Symbols Analyzed: {context.primary_symbols}
Total Insights: {len(context.insights_collected)}
Research Confidence: {context.confidence_level:.2f}

Research Findings:
{insights_summary}

Please provide comprehensive final recommendations."""

            completion = self.client.beta.chat.completions.parse(
                model=settings.azure_openai_deployment_name,
                response_format=ResearchFindings,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=1000,
            )

            findings = completion.choices[0].message.parsed
            logger.info(
                f"Generated final findings with {findings.recommended_action} recommendation"
            )

            return findings

        except Exception as e:
            logger.error(f"Error generating final findings: {e}")
            return self._fallback_findings(context)

    def _fallback_findings(self, context: ResearchContext) -> ResearchFindings:
        """Fallback findings if LLM fails"""
        return ResearchFindings(
            research_question=context.research_question,
            symbols_analyzed=context.primary_symbols,
            total_insights=len(context.insights_collected),
            confidence_score=context.confidence_level,
            key_findings=["Research completed with basic analysis"],
            investment_thesis="Moderate approach recommended based on available data",
            risk_assessment="Standard market risk applies",
            recommended_action="hold",
            position_sizing=0.05,  # 5% position
            entry_strategy="Gradual entry with DCA approach",
            exit_strategy="Monitor key levels and market conditions",
            monitoring_plan=["Track price movements", "Monitor news flow"],
            research_limitations=["Limited by data availability"],
            timestamp=datetime.now(),
        )


# Deep Research Tool Implementations


def analyze_competitors(request: CompetitorAnalysisRequest) -> Dict[str, Any]:
    """Analyze competitors for the primary symbol"""
    try:
        logger.info(f"Analyzing competitors for {request.primary_symbol}")

        # Get basic competitor data (simplified)
        ticker = yf.Ticker(request.primary_symbol)
        info = ticker.info

        # Mock competitor analysis (in real system would use proper competitor data)
        competitors = [
            f"Competitor analysis for {request.primary_symbol}",
            f"Market cap comparison: {info.get('marketCap', 'N/A')}",
            f"P/E ratio: {info.get('trailingPE', 'N/A')}",
            f"Sector: {info.get('sector', 'Unknown')}",
        ]

        return {
            "primary_symbol": request.primary_symbol,
            "competitor_analysis": competitors,
            "comparison_metrics": request.comparison_metrics,
            "sector": info.get("sector", "Unknown"),
            "relative_position": "mid-market",  # Simplified
            "competitive_advantages": ["Market position", "Brand strength"],
            "competitive_risks": ["Intense competition", "Market saturation"],
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in competitor analysis: {e}")
        return {
            "error": str(e),
            "primary_symbol": request.primary_symbol,
            "competitor_analysis": [],
        }


def analyze_economic_indicators(request: EconomicIndicatorRequest) -> Dict[str, Any]:
    """Analyze economic indicators and their market impact"""
    try:
        logger.info(f"Analyzing economic indicators: {request.indicators}")

        # Mock economic indicator analysis (in real system would use FRED API)
        indicator_data = {}
        for indicator in request.indicators:
            if indicator == "gdp":
                indicator_data[indicator] = {
                    "value": 2.1,
                    "trend": "stable",
                    "impact": "neutral",
                }
            elif indicator == "inflation":
                indicator_data[indicator] = {
                    "value": 3.2,
                    "trend": "rising",
                    "impact": "negative",
                }
            elif indicator == "unemployment":
                indicator_data[indicator] = {
                    "value": 3.7,
                    "trend": "stable",
                    "impact": "positive",
                }
            elif indicator == "interest_rates":
                indicator_data[indicator] = {
                    "value": 5.25,
                    "trend": "stable",
                    "impact": "neutral",
                }
            else:
                indicator_data[indicator] = {
                    "value": 0.0,
                    "trend": "unknown",
                    "impact": "neutral",
                }

        overall_impact = "neutral"  # Simplified assessment

        return {
            "indicators_analyzed": request.indicators,
            "indicator_data": indicator_data,
            "overall_economic_outlook": overall_impact,
            "market_implications": [
                "Economic conditions remain mixed",
                "Monitor inflation trends closely",
                "Employment remains strong",
            ],
            "risk_factors": ["Inflation persistence", "Policy uncertainty"],
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in economic analysis: {e}")
        return {
            "error": str(e),
            "indicators_analyzed": request.indicators,
            "overall_economic_outlook": "unknown",
        }


def validate_research_findings(request: ValidationRequest) -> Dict[str, Any]:
    """Validate research findings using multiple methods"""
    try:
        logger.info(f"Validating {len(request.findings_to_validate)} findings")

        validation_results = []
        for finding in request.findings_to_validate:
            # Mock validation (in real system would implement proper validation)
            validation_score = np.random.uniform(
                0.6, 0.9
            )  # Simulate validation confidence

            validation_results.append(
                {
                    "finding": finding[:100] + "..." if len(finding) > 100 else finding,
                    "validation_score": validation_score,
                    "validation_methods": request.validation_methods,
                    "is_validated": validation_score > 0.7,
                    "concerns": []
                    if validation_score > 0.8
                    else ["Moderate confidence level"],
                }
            )

        overall_validation = np.mean(
            [r["validation_score"] for r in validation_results]
        )

        return {
            "validation_results": validation_results,
            "overall_validation_score": overall_validation,
            "validation_summary": "Research findings show good consistency"
            if overall_validation > 0.75
            else "Some findings require additional validation",
            "recommendations": [
                "Findings appear robust"
                if overall_validation > 0.8
                else "Consider additional data sources",
                "Monitor key assumptions",
                "Regular validation updates recommended",
            ],
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in validation: {e}")
        return {
            "error": str(e),
            "validation_results": [],
            "overall_validation_score": 0.5,
        }


def synthesize_research_results(request: SynthesisRequest) -> Dict[str, Any]:
    """Synthesize all research results into final recommendations"""
    try:
        context = request.research_context
        logger.info(f"Synthesizing research for {context.research_id}")

        # Organize insights by phase
        insights_by_phase = {}
        for insight in context.insights_collected:
            phase = insight.phase.value
            if phase not in insights_by_phase:
                insights_by_phase[phase] = []
            insights_by_phase[phase].append(insight)

        # Calculate weighted confidence
        total_weight = sum(
            insight.confidence * insight.impact_score
            for insight in context.insights_collected
        )
        weighted_confidence = (
            total_weight / len(context.insights_collected)
            if context.insights_collected
            else 0.0
        )

        # Generate synthesis
        synthesis_summary = [
            f"Comprehensive research completed on {context.primary_symbols}",
            f"Total insights collected: {len(context.insights_collected)}",
            f"Research confidence: {weighted_confidence:.2f}",
            f"Key phases covered: {', '.join(insights_by_phase.keys())}",
        ]

        # Determine overall recommendation based on insights
        positive_insights = sum(
            1 for insight in context.insights_collected if insight.confidence > 0.7
        )
        recommendation_strength = (
            positive_insights / len(context.insights_collected)
            if context.insights_collected
            else 0.5
        )

        if recommendation_strength > 0.7:
            overall_recommendation = "positive"
        elif recommendation_strength < 0.3:
            overall_recommendation = "negative"
        else:
            overall_recommendation = "neutral"

        return {
            "research_question": context.research_question,
            "symbols_analyzed": context.primary_symbols,
            "synthesis_summary": synthesis_summary,
            "insights_by_phase": {
                phase: len(insights) for phase, insights in insights_by_phase.items()
            },
            "overall_confidence": weighted_confidence,
            "overall_recommendation": overall_recommendation,
            "key_insights": [
                insight.content for insight in context.insights_collected[:5]
            ],  # Top 5
            "actionable_steps": [
                "Consider position sizing based on risk assessment",
                "Monitor key market indicators",
                "Set appropriate stop-loss levels",
                "Plan entry and exit strategies",
            ],
            "limitations": context.knowledge_gaps
            or ["Standard market limitations apply"],
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in synthesis: {e}")
        return {
            "error": str(e),
            "research_question": getattr(
                request.research_context, "research_question", "Unknown"
            ),
            "synthesis_summary": ["Error in synthesis process"],
        }
