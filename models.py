#!/usr/bin/env python3
"""
Pydantic models for Financial Trading Agent

Data models for market data, trading signals, forecasts, and agent responses.
Based on the SGR pattern architecture.
"""

from typing import List, Union, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ============ Market Data Models ============


class MarketData(BaseModel):
    """Market data for a specific symbol"""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    market_cap: Optional[float] = None
    volatility: Optional[float] = None

    class Config:
        extra = "forbid"


class NewsSentiment(BaseModel):
    """News sentiment analysis result"""

    headline: str
    content: str
    source: str
    timestamp: datetime
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)  # -1 to 1
    relevance_score: float = Field(..., ge=0.0, le=1.0)  # 0 to 1
    impact_assessment: str = Field(
        description="Expected market impact (low/medium/high)"
    )

    class Config:
        extra = "forbid"


class EconomicIndicator(BaseModel):
    """Economic indicator data"""

    indicator_name: str
    value: float
    unit: str
    timestamp: datetime
    source: str = "FRED"
    previous_value: Optional[float] = None
    forecast_value: Optional[float] = None

    class Config:
        extra = "forbid"


# ============ Analysis Results Models ============


class MarketDataResponse(BaseModel):
    """Response from market data collection"""

    symbols_analyzed: List[str]
    market_data: List[MarketData]
    market_trend: str = Field(
        description="Overall market trend (bullish/bearish/neutral)"
    )
    volatility_assessment: str = Field(description="Market volatility level")
    key_insights: List[str] = Field(description="Key market insights")
    timestamp: datetime

    class Config:
        extra = "forbid"


class ForecastResult(BaseModel):
    """Probabilistic forecast result from SGR agents"""

    question: str
    prediction_probability: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: Dict[str, float] = Field(
        description="Confidence intervals (e.g., {'80%': 0.1, '95%': 0.2})"
    )
    rationale: str
    forecast_horizon: str = Field(description="Time horizon (1_day, 1_week, 1_month)")
    agent_type: str = Field(description="Agent type (bullish, bearish, technical)")
    timestamp: datetime
    brier_score: Optional[float] = None

    class Config:
        extra = "forbid"


class TradingSignal(BaseModel):
    """Trading signal recommendation"""

    symbol: str
    signal_type: Literal["buy", "sell", "hold"]
    strength: float = Field(..., ge=0.0, le=1.0)  # 0 to 1
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    timestamp: datetime

    class Config:
        extra = "forbid"


class RiskAssessment(BaseModel):
    """Risk assessment for trade or portfolio"""

    symbol: Optional[str] = None
    portfolio_symbols: Optional[List[str]] = None
    var_1d: float = Field(description="1-day Value at Risk")
    var_5d: float = Field(description="5-day Value at Risk")
    max_drawdown: float = Field(description="Maximum drawdown estimate")
    volatility: float = Field(description="Annualized volatility")
    sharpe_ratio: Optional[float] = None
    beta: Optional[float] = None
    risk_level: Literal["low", "medium", "high"]
    recommendations: List[str]
    timestamp: datetime

    class Config:
        extra = "forbid"


class TradingRecommendation(BaseModel):
    """Complete trading recommendation"""

    symbol: str
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = Field(description="Recommended position size")
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    risk_assessment: RiskAssessment
    timestamp: datetime

    class Config:
        extra = "forbid"


# ============ Portfolio Models ============


class Trade(BaseModel):
    """Individual trade record"""

    trade_id: str
    symbol: str
    action: Literal["buy", "sell"]
    quantity: float
    price: float
    timestamp: datetime
    order_type: str = "market"
    status: str = "executed"

    class Config:
        extra = "forbid"


class PortfolioPosition(BaseModel):
    """Current portfolio position"""

    symbol: str
    quantity: float
    avg_cost: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    weight: Optional[float] = None
    last_updated: datetime

    class Config:
        extra = "forbid"


class PerformanceReport(BaseModel):
    """Portfolio performance report"""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    report_period: str
    timestamp: datetime

    class Config:
        extra = "forbid"


# ============ SGR Trading Request Models ============


class MarketDataRequest(BaseModel):
    """Request market data for specific symbols"""

    tool: Literal["get_market_data"]
    symbols: List[str]
    timeframe: str = Field(default="1d", description="Data timeframe (1d, 5m, 1h)")
    period: str = Field(default="1mo", description="Historical period (1mo, 3mo, 1y)")

    class Config:
        extra = "forbid"


class TradingAnalysisRequest(BaseModel):
    """Request comprehensive trading analysis"""

    tool: Literal["analyze_trading_opportunity"]
    symbol: str
    analysis_type: str = Field(default="comprehensive", description="Type of analysis")
    budget: Optional[float] = Field(default=None, description="Investment budget")
    risk_tolerance: str = Field(default="medium", description="Risk tolerance level")

    class Config:
        extra = "forbid"


class ForecastRequest(BaseModel):
    """Request probabilistic forecast"""

    tool: Literal["generate_forecast"]
    question: str = Field(description="Forecast question to answer")
    symbols: List[str] = Field(description="Symbols to analyze for forecast")
    forecast_horizon: str = Field(default="1_week", description="Forecast time horizon")
    agent_types: List[str] = Field(
        default=["bullish", "bearish", "technical"], description="Agent types to use"
    )

    class Config:
        extra = "forbid"


class RiskAssessmentRequest(BaseModel):
    """Request risk assessment"""

    tool: Literal["assess_risk"]
    symbol: Optional[str] = None
    trade_amount: Optional[float] = None
    assessment_type: str = Field(
        default="portfolio", description="Type of risk assessment"
    )

    class Config:
        extra = "forbid"


class NewsAnalysisRequest(BaseModel):
    """Request news sentiment analysis"""

    tool: Literal["analyze_news"]
    symbols: List[str]
    sources: List[str] = Field(
        default=["reuters", "bloomberg"], description="News sources"
    )
    lookback_hours: int = Field(default=24, description="Hours to look back for news")

    class Config:
        extra = "forbid"


class BacktestRequest(BaseModel):
    """Request strategy backtesting"""

    tool: Literal["run_backtest"]
    strategy_name: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = Field(default=100000, description="Starting capital")

    class Config:
        extra = "forbid"


class ReportTaskCompletion(BaseModel):
    """Report completion of SGR trading analysis"""

    tool: Literal["report_completion"]
    code: str = Field(description="Completion status code")
    completed_steps_laconic: List[str] = Field(
        description="Laconic list of completed analysis steps"
    )
    final_recommendation: Optional[str] = Field(
        default=None, description="Final trading recommendation"
    )

    class Config:
        extra = "forbid"


# ============ Web Scraping & Intelligence Models ============


class WebAnalysisRequest(BaseModel):
    """Request for web-based financial analysis"""

    tool: Literal["analyze_web_content"]
    url: str = Field(description="URL to analyze for financial content")
    analysis_type: str = Field(
        default="financial_news",
        description="Type of analysis: financial_news, earnings_report, market_analysis",
    )
    extract_data_points: List[str] = Field(
        default=["sentiment", "key_metrics", "price_targets"],
        description="Data points to extract from the content",
    )
    include_links: bool = Field(
        default=True, description="Include related links in analysis"
    )

    class Config:
        extra = "forbid"


class ComprehensiveWebResearch(BaseModel):
    """Comprehensive web research for financial topics"""

    tool: Literal["research_financial_topic"]
    search_query: str = Field(description="Search query for financial research")
    research_depth: str = Field(
        default="comprehensive",
        description="Research depth: basic, comprehensive, deep",
    )
    max_sources: int = Field(
        default=5,
        description="Maximum number of sources to analyze (limited to 5 for performance)",
    )
    include_news: bool = Field(
        default=True, description="Include news sources in research"
    )
    include_analyst_reports: bool = Field(
        default=True, description="Include analyst reports in research"
    )
    extract_financial_data: bool = Field(
        default=True, description="Extract financial metrics and data"
    )
    time_range: str = Field(
        default="1_week", description="Time range for research: 1_day, 1_week, 1_month"
    )

    class Config:
        extra = "forbid"


class WebContentAnalysis(BaseModel):
    """Analyzed web content result"""

    url: str
    title: str
    content_type: str  # news, report, analysis, filing
    sentiment_score: float  # -1.0 to 1.0
    key_financial_metrics: Dict[str, Any]
    extracted_data: Dict[str, Any]
    market_impact_assessment: str
    credibility_score: float  # 0.0 to 1.0
    timestamp: datetime
    source_quality: str  # high, medium, low


class ResearchSummary(BaseModel):
    """Summary of web research results"""

    search_query: str
    sources_analyzed: int
    overall_sentiment: float
    key_findings: List[str]
    financial_consensus: Dict[str, Any]
    risk_factors: List[str]
    opportunities: List[str]
    source_quality_distribution: Dict[str, int]
    timestamp: datetime


# ============ SGR Memory Management Tools ============


class TradingRuleParameters(BaseModel):
    """Trading rule parameters with known fields"""

    max_position_size: Optional[float] = None
    max_risk_per_trade: Optional[float] = None
    stop_loss_required: Optional[bool] = None
    position_sizing_method: Optional[str] = None
    max_correlation: Optional[float] = None
    sectors: Optional[List[str]] = None
    min_market_cap: Optional[float] = None
    market_condition: Optional[str] = None
    exclude_penny_stocks: Optional[bool] = None
    timeframes: Optional[List[str]] = None
    indicators: Optional[List[str]] = None
    confirmation_required: Optional[bool] = None
    volume_threshold: Optional[float] = None
    divergence_check: Optional[bool] = None
    check_frequency: Optional[str] = None

    class Config:
        extra = "forbid"


class ChatMessageMetadata(BaseModel):
    """Chat message metadata with known fields"""

    symbol: Optional[str] = None
    task_type: Optional[str] = None
    analysis_type: Optional[str] = None
    message_type: Optional[str] = None
    steps_completed: Optional[int] = None
    task: Optional[str] = None

    class Config:
        extra = "forbid"


class MemoryFilterCriteria(BaseModel):
    """Memory filter criteria with known fields"""

    rule_type: Optional[str] = None
    active: Optional[bool] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    class Config:
        extra = "forbid"


class CreateTradingRule(BaseModel):
    """Create a trading rule for agent memory (SGR style)"""

    tool: Literal["create_trading_rule"]
    rule_description: str = Field(
        description="Human-readable description of the trading rule"
    )
    rule_type: str = Field(
        default="general",
        description="Type: risk_management, trading_preference, analysis_guideline",
    )
    parameters: TradingRuleParameters = Field(
        default_factory=TradingRuleParameters, description="Rule parameters and values"
    )
    priority: int = Field(
        default=1, ge=1, le=10, description="Rule priority (1-10, 10 is highest)"
    )

    class Config:
        extra = "forbid"


class GetTradingMemory(BaseModel):
    """Retrieve trading memory and rules"""

    tool: Literal["get_trading_memory"]
    memory_type: str = Field(
        default="all", description="Type: all, rules, market_data, forecasts, trades"
    )
    filter_by: Optional[MemoryFilterCriteria] = Field(
        default=None, description="Filter criteria"
    )

    class Config:
        extra = "forbid"


class UpdateTradingRule(BaseModel):
    """Update existing trading rule"""

    tool: Literal["update_trading_rule"]
    rule_id: str = Field(description="ID of rule to update")
    new_description: Optional[str] = Field(
        default=None, description="New rule description"
    )
    new_parameters: Optional[TradingRuleParameters] = Field(
        default=None, description="New rule parameters"
    )
    new_priority: Optional[int] = Field(
        default=None, ge=1, le=10, description="New priority level"
    )

    class Config:
        extra = "forbid"


class DeleteTradingRule(BaseModel):
    """Delete trading rule from memory"""

    tool: Literal["delete_trading_rule"]
    rule_id: str = Field(description="ID of rule to delete")
    reason: str = Field(description="Reason for deletion")

    class Config:
        extra = "forbid"


class CreateChatSession(BaseModel):
    """Create new chat session in memory"""

    tool: Literal["create_chat_session"]
    session_name: str = Field(description="Human-readable name for the chat session")
    user_id: str = Field(default="default", description="User identifier")
    session_type: str = Field(
        default="trading_analysis", description="Type of chat session"
    )

    class Config:
        extra = "forbid"


class SaveChatMessage(BaseModel):
    """Save chat message to session memory"""

    tool: Literal["save_chat_message"]
    session_id: str = Field(description="Chat session ID")
    message_type: str = Field(description="Type: user, assistant, system, tool")
    content: str = Field(description="Message content")
    metadata: Optional[ChatMessageMetadata] = Field(
        default=None, description="Additional message metadata"
    )

    class Config:
        extra = "forbid"


class GetChatHistory(BaseModel):
    """Retrieve chat history from memory"""

    tool: Literal["get_chat_history"]
    session_id: Optional[str] = Field(
        default=None, description="Specific session ID or None for all sessions"
    )
    limit: int = Field(default=50, description="Maximum number of messages to retrieve")
    message_types: List[str] = Field(
        default=["user", "assistant"], description="Types of messages to include"
    )

    class Config:
        extra = "forbid"


# ============ Alpha Research Models ============


class AlphaOp(BaseModel):
    """Alpha operation specification (finding alphas operators)"""

    name: Literal[
        "delta", "delay", "ts_mean", "ts_std", "zscore", "ts_rank", "decay_linear"
    ]
    window: Optional[int] = None
    k: Optional[int] = None  # Step for delta/delay operations

    class Config:
        extra = "forbid"


class AlphaSpec(BaseModel):
    """Alpha factor specification"""

    name: str
    input: Literal["open", "high", "low", "close", "volume", "returns"]
    ops: List[AlphaOp] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class AlphaPoint(BaseModel):
    """Single alpha factor data point"""

    timestamp: datetime
    value: float

    class Config:
        extra = "forbid"


class AlphaSeries(BaseModel):
    """Alpha factor time series for a symbol"""

    symbol: str
    factor: str
    points: List[AlphaPoint]

    class Config:
        extra = "forbid"


class AlphaReport(BaseModel):
    """Alpha factor analysis report"""

    factor: str
    last_value: Dict[str, float]  # Last value per symbol
    ic1d: Optional[float] = None  # Information coefficient (1-day)
    coverage: int  # Number of observations

    class Config:
        extra = "forbid"


class AlphaGenerationRequest(BaseModel):
    """Request to generate alpha factors"""

    tool: Literal["generate_alphas"]
    symbols: List[str]
    specs: List[AlphaSpec]
    timeframe: str = Field(default="1d")
    period: str = Field(default="3mo")

    class Config:
        extra = "forbid"


# ============ SGR Response Model ============


class SGRTradingResponse(BaseModel):
    """Schema-Guided Reasoning response for financial trading"""

    current_state: str = Field(
        max_length=200,
        description="Current analysis state (what the agent is thinking about)",
    )
    plan_remaining_steps_brief: List[str] = Field(
        max_items=5,
        description="Brief list of remaining analysis steps to complete the task (max 5 items)",
    )
    function: Union[
        MarketDataRequest,
        TradingAnalysisRequest,
        ForecastRequest,
        RiskAssessmentRequest,
        NewsAnalysisRequest,
        BacktestRequest,
        WebAnalysisRequest,
        ComprehensiveWebResearch,
        AlphaGenerationRequest,  # Alpha Factory tool
        ReportTaskCompletion,
        # Memory tools temporarily disabled for debugging
        CreateTradingRule,
        GetTradingMemory,
        UpdateTradingRule,
        DeleteTradingRule,
        CreateChatSession,
        SaveChatMessage,
        GetChatHistory,
        # Deep Research tools would be imported separately when needed
    ] = Field(description="Next financial analysis tool to execute")
    task_completed: bool = Field(
        default=False, description="Whether the financial analysis is complete"
    )

    class Config:
        extra = "forbid"


# ============ Memory and Context Models ============


class TradingMemory(BaseModel):
    """Agent memory for trading context"""

    market_data: List[MarketData] = Field(default_factory=list)
    forecasts: List[ForecastResult] = Field(default_factory=list)
    positions: List[PortfolioPosition] = Field(default_factory=list)
    recent_trades: List[Trade] = Field(default_factory=list)
    risk_assessments: List[RiskAssessment] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class CustomerRule(BaseModel):
    """Customer-specific trading rules and preferences"""

    rule_id: str
    customer_id: str
    rule_type: str = Field(description="Type of rule (risk_limit, preference, etc.)")
    rule_value: Dict[str, Any] = Field(description="Rule parameters and values")
    active: bool = True
    created_at: datetime
    last_modified: datetime

    class Config:
        extra = "forbid"
