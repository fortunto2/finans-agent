#!/usr/bin/env python3
"""
Deep Research Models for SGR Financial Trading Agent

Adaptive planning and reasoning models based on SGR-classic pattern.
Enhanced for financial market analysis with deep research capabilities.
"""

from typing import List, Union, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ResearchPhase(str, Enum):
    """Deep research phases for financial analysis"""

    MARKET_DISCOVERY = "market_discovery"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    FORECAST_GENERATION = "forecast_generation"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"


class ResearchAction(str, Enum):
    """Available research actions"""

    COLLECT_MARKET_DATA = "collect_market_data"
    ANALYZE_FUNDAMENTALS = "analyze_fundamentals"
    TECHNICAL_INDICATORS = "technical_indicators"
    NEWS_SENTIMENT = "news_sentiment"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    ECONOMIC_INDICATORS = "economic_indicators"
    RISK_METRICS = "risk_metrics"
    FORECAST_MODELS = "forecast_models"
    PORTFOLIO_ALLOCATION = "portfolio_allocation"
    BACKTESTING = "backtesting"
    VALIDATE_FINDINGS = "validate_findings"
    SYNTHESIZE_RESULTS = "synthesize_results"


class ResearchInsight(BaseModel):
    """Individual research insight"""

    insight_id: str
    phase: ResearchPhase
    action: ResearchAction
    content: str = Field(description="The actual insight or finding")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this insight"
    )
    impact_score: float = Field(
        ..., ge=0.0, le=1.0, description="Expected impact on final recommendation"
    )
    sources: List[str] = Field(description="Data sources used for this insight")
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class ResearchContext(BaseModel):
    """Current research context and accumulated knowledge"""

    research_id: str
    primary_symbols: List[str] = Field(description="Primary symbols being researched")
    research_question: str = Field(description="Main research question or objective")
    current_phase: ResearchPhase
    insights_collected: List[ResearchInsight] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(
        default_factory=list, description="Identified gaps in analysis"
    )
    confidence_level: float = Field(default=0.0, ge=0.0, le=1.0)
    research_depth: int = Field(
        default=0, description="Current depth of research (number of iterations)"
    )
    max_depth: int = Field(default=10, description="Maximum research depth allowed")
    priority_areas: List[str] = Field(
        default_factory=list, description="Areas requiring priority attention"
    )

    class Config:
        extra = "forbid"


class NextStep(BaseModel):
    """SGR Core - Determines next reasoning step with adaptive planning"""

    reasoning: str = Field(description="Detailed reasoning for choosing this next step")
    current_understanding: str = Field(
        description="Summary of current market understanding"
    )
    knowledge_gaps: List[str] = Field(
        description="Identified gaps that need to be filled"
    )
    next_action: ResearchAction = Field(description="The specific action to take next")
    action_rationale: str = Field(description="Why this specific action was chosen")
    expected_insights: List[str] = Field(description="What insights we expect to gain")
    research_priority: Literal["critical", "high", "medium", "low"] = Field(
        description="Priority level of this research step"
    )
    estimated_completion_progress: float = Field(
        ..., ge=0.0, le=1.0, description="Estimated progress after this step"
    )
    adaptive_plan: List[str] = Field(description="Remaining steps in adaptive plan")
    should_continue: bool = Field(
        description="Whether research should continue after this step"
    )

    class Config:
        extra = "forbid"


class DeepResearchRequest(BaseModel):
    """Request to start deep research analysis"""

    tool: Literal["deep_research"]
    research_question: str = Field(
        description="The main research question or investment thesis to investigate"
    )
    symbols: List[str] = Field(description="Symbols to research in depth")
    research_scope: Literal["comprehensive", "focused", "rapid"] = Field(
        default="comprehensive", description="Scope of research analysis"
    )
    max_depth: int = Field(
        default=15, description="Maximum research iterations allowed"
    )
    focus_areas: Optional[List[ResearchPhase]] = Field(
        default=None,
        description="Specific areas to focus on (if None, will cover all areas)",
    )
    budget_constraint: Optional[float] = Field(
        default=None, description="Budget constraint for analysis"
    )
    time_horizon: str = Field(
        default="medium_term", description="Investment time horizon"
    )

    class Config:
        extra = "forbid"


class ResearchFindings(BaseModel):
    """Final research findings and recommendations"""

    research_question: str
    symbols_analyzed: List[str]
    total_insights: int
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    key_findings: List[str] = Field(description="Most important research findings")
    investment_thesis: str = Field(
        description="Overall investment thesis based on research"
    )
    risk_assessment: str = Field(description="Comprehensive risk assessment")
    recommended_action: Literal[
        "strong_buy", "buy", "hold", "sell", "strong_sell", "avoid"
    ]
    position_sizing: Optional[float] = Field(
        description="Recommended position size as % of portfolio"
    )
    entry_strategy: str = Field(description="Recommended entry strategy")
    exit_strategy: str = Field(description="Recommended exit strategy")
    monitoring_plan: List[str] = Field(description="Key metrics and events to monitor")
    research_limitations: List[str] = Field(
        description="Limitations and assumptions in the research"
    )
    timestamp: datetime

    class Config:
        extra = "forbid"


class DeepResearchResponse(BaseModel):
    """Response from deep research system"""

    current_step: int
    total_steps_planned: int
    research_context: ResearchContext
    next_step: Optional[NextStep] = None
    findings: Optional[ResearchFindings] = None
    is_complete: bool = Field(default=False)
    session_summary: str = Field(description="Summary of current research session")

    class Config:
        extra = "forbid"


# Research Tool Request Models (extending existing models)


class CompetitorAnalysisRequest(BaseModel):
    """Request competitor analysis"""

    tool: Literal["analyze_competitors"]
    primary_symbol: str
    sector: Optional[str] = None
    comparison_metrics: List[str] = Field(
        default=["market_cap", "pe_ratio", "revenue_growth", "profit_margins"],
        description="Metrics to compare",
    )

    class Config:
        extra = "forbid"


class EconomicIndicatorRequest(BaseModel):
    """Request economic indicator analysis"""

    tool: Literal["analyze_economic_indicators"]
    indicators: List[str] = Field(
        default=["gdp", "inflation", "unemployment", "interest_rates"],
        description="Economic indicators to analyze",
    )
    impact_assessment: bool = Field(
        default=True, description="Assess impact on target symbols"
    )

    class Config:
        extra = "forbid"


class ValidationRequest(BaseModel):
    """Request validation of research findings"""

    tool: Literal["validate_research"]
    findings_to_validate: List[str]
    validation_methods: List[str] = Field(
        default=["cross_reference", "historical_analysis", "peer_review"],
        description="Validation methods to use",
    )

    class Config:
        extra = "forbid"


class SynthesisRequest(BaseModel):
    """Request synthesis of all research findings"""

    tool: Literal["synthesize_research"]
    research_context: ResearchContext
    focus_on_actionability: bool = Field(
        default=True, description="Focus on actionable insights"
    )

    class Config:
        extra = "forbid"


# Updated SGR Response for Deep Research


class SGRDeepResearchResponse(BaseModel):
    """SGR response specifically for deep research mode"""

    current_state: str = Field(description="Current research state and progress")
    research_phase: ResearchPhase = Field(description="Current phase of research")
    insights_summary: str = Field(description="Summary of insights gathered so far")
    next_step: NextStep = Field(description="Adaptive next step planning")
    function: Union[
        # Existing tools
        "MarketDataRequest",
        "TradingAnalysisRequest",
        "ForecastRequest",
        "RiskAssessmentRequest",
        "NewsAnalysisRequest",
        "BacktestRequest",
        # New deep research tools
        "CompetitorAnalysisRequest",
        "EconomicIndicatorRequest",
        "ValidationRequest",
        "SynthesisRequest",
        "DeepResearchRequest",
        "ReportTaskCompletion",
    ] = Field(description="Next research tool to execute")
    research_depth: int = Field(description="Current research depth level")
    confidence_progression: List[float] = Field(
        description="Confidence scores over time"
    )

    class Config:
        extra = "forbid"
