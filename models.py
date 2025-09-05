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
    impact_assessment: str = Field(description="Expected market impact (low/medium/high)")

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
    market_trend: str = Field(description="Overall market trend (bullish/bearish/neutral)")
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
    assessment_type: str = Field(default="portfolio", description="Type of risk assessment")

    class Config:
        extra = "forbid"


class NewsAnalysisRequest(BaseModel):
    """Request news sentiment analysis"""

    tool: Literal["analyze_news"]
    symbols: List[str]
    sources: List[str] = Field(default=["reuters", "bloomberg"], description="News sources")
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


# ============ SGR Response Model ============


class SGRTradingResponse(BaseModel):
    """Schema-Guided Reasoning response for financial trading"""

    current_state: str = Field(
        description="Current analysis state (what the agent is thinking about)"
    )
    plan_remaining_steps_brief: List[str] = Field(
        description="Brief list of remaining analysis steps to complete the task"
    )
    function: Union[
        MarketDataRequest,
        TradingAnalysisRequest,
        ForecastRequest,
        RiskAssessmentRequest,
        NewsAnalysisRequest,
        BacktestRequest,
        ReportTaskCompletion,
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