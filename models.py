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


class Config:
    extra = "forbid"

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


class Config:
    extra = "forbid"

    headline: str
    content: str
    source: str
    timestamp: datetime
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)  # -1 to 1
    relevance_score: float = Field(..., ge=0.0, le=1.0)  # 0 to 1
    entities_mentioned: List[str] = []
    impact_assessment: Optional[str] = None


class EconomicIndicator(BaseModel):
    """Economic indicator data"""


class Config:
    extra = "forbid"

    indicator_name: str
    value: float
    forecast: Optional[float] = None
    previous: Optional[float] = None
    timestamp: datetime
    impact_level: Optional[str] = None  # "low", "medium", "high"


# ============ Forecast Models ============


class ForecastResult(BaseModel):
    """AI forecast result with probability distribution"""


class Config:
    extra = "forbid"

    question: str
    prediction_probability: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: Dict[str, float] = {}  # e.g., {"95%": 0.15}
    rationale: str
    forecast_horizon: str  # e.g., "1_week", "1_month"
    agent_type: str  # e.g., "bullish", "bearish", "technical"
    timestamp: datetime
    brier_score: Optional[float] = None  # Will be calculated after outcome


class ForecastConsensus(BaseModel):
    """Aggregated forecast from multiple agents"""


class Config:
    extra = "forbid"

    question: str
    consensus_probability: float = Field(..., ge=0.0, le=1.0)
    individual_forecasts: List[ForecastResult] = []
    disagreement_level: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime


# ============ Trading Models ============


class TradingSignal(BaseModel):
    """Trading signal from analysis"""


class Config:
    extra = "forbid"

    symbol: str
    signal_type: Literal["buy", "sell", "hold"]
    signal_strength: float = Field(..., ge=0.0, le=1.0)
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_level: str  # "low", "medium", "high"
    reasoning: str
    generated_by: str  # Agent that generated signal
    timestamp: datetime


class Trade(BaseModel):
    """Executed trade record"""


class Config:
    extra = "forbid"

    trade_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    timestamp: datetime
    status: Literal["pending", "executed", "cancelled", "failed"]
    commission: Optional[float] = None
    pnl: Optional[float] = None  # Profit/Loss when closed


class PortfolioPosition(BaseModel):
    """Current portfolio position"""


class Config:
    extra = "forbid"

    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float  # Portfolio weight percentage
    last_updated: datetime


# ============ Risk Management Models ============


class RiskAssessment(BaseModel):
    """Risk assessment for a trade or portfolio"""


class Config:
    extra = "forbid"

    symbol: Optional[str] = None  # None for portfolio-level risk
    var_1d: float  # 1-day Value at Risk
    var_5d: float  # 5-day Value at Risk
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    volatility: float
    risk_level: Literal["low", "medium", "high", "extreme"]
    recommendations: List[str] = []
    timestamp: datetime


class ComplianceCheck(BaseModel):
    """Compliance validation result"""


class Config:
    extra = "forbid"

    check_type: str
    symbol: Optional[str] = None
    trade_amount: Optional[float] = None
    is_compliant: bool
    violations: List[str] = []
    regulatory_notes: List[str] = []
    timestamp: datetime


# ============ SGR Tool Models ============


class MarketDataRequest(BaseModel):
    """Request for market data"""


class Config:
    extra = "forbid"

    tool: Literal["get_market_data"]
    symbols: List[str]
    timeframe: str = "1d"  # 1m, 5m, 1h, 1d, 1w
    period: str = "1mo"  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max


class ForecastRequest(BaseModel):
    """Request for market forecast"""


class Config:
    extra = "forbid"

    tool: Literal["generate_forecast"]
    question: str
    symbols: List[str] = []
    forecast_horizon: str = "1_week"
    agent_types: List[str] = ["bullish", "bearish", "technical"]


class TradingAnalysisRequest(BaseModel):
    """Request for trading analysis"""


class Config:
    extra = "forbid"

    tool: Literal["analyze_trading_opportunity"]
    symbol: str
    analysis_type: str = "comprehensive"  # "technical", "fundamental", "comprehensive"
    budget: Optional[float] = None
    risk_tolerance: str = "medium"  # "low", "medium", "high"


class BacktestRequest(BaseModel):
    """Request for strategy backtesting"""


class Config:
    extra = "forbid"

    tool: Literal["run_backtest"]
    strategy_name: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000


class RiskAssessmentRequest(BaseModel):
    """Request for risk assessment"""


class Config:
    extra = "forbid"

    tool: Literal["assess_risk"]
    symbol: Optional[str] = None
    trade_amount: Optional[float] = None
    assessment_type: str = "portfolio"  # "trade", "portfolio"


class NewsAnalysisRequest(BaseModel):
    """Request for news sentiment analysis"""


class Config:
    extra = "forbid"

    tool: Literal["analyze_news"]
    symbols: List[str] = []
    sources: List[str] = ["reuters", "bloomberg", "cnbc"]
    lookback_hours: int = 24


class ProphetArenaRequest(BaseModel):
    """Request for Prophet Arena forecast validation"""


class Config:
    extra = "forbid"

    tool: Literal["submit_to_prophet_arena"]
    question: str
    prediction_probability: float = Field(..., ge=0.0, le=1.0)
    forecast_horizon: str


class ComplianceCheckRequest(BaseModel):
    """Request for compliance check"""


class Config:
    extra = "forbid"

    tool: Literal["check_compliance"]
    symbol: str
    trade_amount: float
    check_types: List[str] = ["position_limits", "regulatory", "risk_limits"]


class GenerateReportRequest(BaseModel):
    """Request to generate trading report"""


class Config:
    extra = "forbid"

    tool: Literal["generate_report"]
    report_type: str  # "portfolio", "performance", "risk", "forecast_accuracy"
    period: str = "1mo"
    include_charts: bool = True


class ReportTaskCompletion(BaseModel):
    """Report task completion with final results"""


class Config:
    extra = "forbid"

    tool: Literal["report_completion"]
    completed_steps_laconic: List[str]
    final_recommendation: str = ""
    code: str = "completed"


# ============ SGR Response Model ============


class SGRTradingResponse(BaseModel):
    """SGR Trading Agent Response with structured reasoning"""


class Config:
    extra = "forbid"

    current_state: str = Field(
        ..., description="Current understanding of the market situation"
    )
    plan_remaining_steps_brief: List[str] = Field(
        ..., description="List of 1-5 remaining steps to complete the analysis"
    )
    task_completed: bool = Field(
        False, description="Whether the task is fully completed"
    )
    function: Union[
        MarketDataRequest,
        ForecastRequest,
        TradingAnalysisRequest,
        BacktestRequest,
        RiskAssessmentRequest,
        NewsAnalysisRequest,
        ProphetArenaRequest,
        ComplianceCheckRequest,
        GenerateReportRequest,
        ReportTaskCompletion,
    ] = Field(..., description="Next tool to execute")

    class Config:
        extra = "forbid"


# ============ Response Models ============


class MarketDataResponse(BaseModel):
    """Market data analysis response"""


class Config:
    extra = "forbid"

    symbols_analyzed: List[str]
    market_data: List[MarketData]
    key_insights: List[str] = []
    market_trend: str  # "bullish", "bearish", "sideways"
    volatility_assessment: str  # "low", "normal", "high", "extreme"
    timestamp: datetime


class TradingRecommendation(BaseModel):
    """Trading recommendation response"""


class Config:
    extra = "forbid"

    symbol: str
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: float  # Percentage of portfolio
    rationale: str
    risk_assessment: RiskAssessment
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime


class PerformanceReport(BaseModel):
    """Trading performance report"""


class Config:
    extra = "forbid"

    period_start: datetime
    period_end: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    current_positions: List[PortfolioPosition] = []
    forecast_accuracy: Optional[float] = None  # Brier score if available


# ============ Database/Memory Models ============


class TradingMemory(BaseModel):
    """In-memory trading database"""


class Config:
    extra = "forbid"

    market_data: List[MarketData] = []
    forecasts: List[ForecastResult] = []
    trades: List[Trade] = []
    positions: List[PortfolioPosition] = []
    news_sentiment: List[NewsSentiment] = []
    risk_assessments: List[RiskAssessment] = []
    performance_history: List[PerformanceReport] = []


class CustomerRule(BaseModel):
    """Customer-specific trading rule"""


class Config:
    extra = "forbid"

    rule_id: str
    customer_context: str
    rule_description: str
    rule_type: str = "trading_preference"
    parameters: Dict[str, Any] = {}
    created_at: datetime
    is_active: bool = True
