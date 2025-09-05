#!/usr/bin/env python3
"""
Schema-Guided Reasoning Financial Trading Agent

Multi-agent AI system for financial forecasting and algorithmic trading.
Based on the SGR pattern from https://abdullin.com/schema-guided-reasoning/demo
Adapted for financial markets with Azure OpenAI integration.
"""

import json
import uuid
from typing import List, Union, Dict, Any
from openai import AzureOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
import logging
from datetime import datetime
import numpy as np

# Import settings and models
from settings import settings, validate_required_keys
from models import (
    MarketDataRequest,
    ForecastRequest,
    TradingAnalysisRequest,
    BacktestRequest,
    RiskAssessmentRequest,
    NewsAnalysisRequest,
    ReportTaskCompletion,
    SGRTradingResponse,
    TradingMemory,
    MarketData,
    ForecastResult,
    TradingRecommendation,
)

# Import market data tools
from market_data_tools import (
    MarketDataCollector,
    TradingAnalyzer,
    NewsAnalyzer,
    RiskAnalyzer,
)

# Setup rich console for beautiful output
console = Console()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialTradingDB:
    """In-memory database for financial trading data"""

    def __init__(self):
        self.memory = TradingMemory()
        self.forecasts_history = []
        self.analysis_history = []
        self.customer_rules = []

        logger.info("Financial Trading Agent initialized with SGR pattern")

    def add_market_data(self, market_data: List[MarketData]):
        """Add market data to memory"""
        self.memory.market_data.extend(market_data)
        # Keep only last 1000 records to manage memory
        if len(self.memory.market_data) > 1000:
            self.memory.market_data = self.memory.market_data[-1000:]

    def add_forecast(self, forecast: ForecastResult):
        """Add forecast to memory"""
        self.memory.forecasts.append(forecast)
        self.forecasts_history.append(
            {"forecast": forecast.model_dump(), "timestamp": datetime.now().isoformat()}
        )

    def get_recent_analysis(self, symbol: str = None, limit: int = 5) -> List[Dict]:
        """Get recent analysis for symbol or all"""
        if symbol:
            relevant = [a for a in self.analysis_history if symbol in str(a)]
            return relevant[-limit:] if relevant else []
        return self.analysis_history[-limit:]


# Global database instance
DB = FinancialTradingDB()

# Initialize market tools
market_collector = MarketDataCollector()
trading_analyzer = TradingAnalyzer()
news_analyzer = NewsAnalyzer()
risk_analyzer = RiskAnalyzer()


# ============ Tool Dispatch Implementation ============


def dispatch(cmd) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Execute SGR financial tools and return results"""

    if isinstance(cmd, MarketDataRequest):
        return get_market_data(cmd.symbols, cmd.timeframe, cmd.period)

    elif isinstance(cmd, TradingAnalysisRequest):
        return analyze_trading_opportunity(
            cmd.symbol, cmd.analysis_type, cmd.budget, cmd.risk_tolerance
        )

    elif isinstance(cmd, ForecastRequest):
        return generate_forecast(
            cmd.question, cmd.symbols, cmd.forecast_horizon, cmd.agent_types
        )

    elif isinstance(cmd, RiskAssessmentRequest):
        return assess_risk(cmd.symbol, cmd.trade_amount, cmd.assessment_type)

    elif isinstance(cmd, NewsAnalysisRequest):
        return analyze_news(cmd.symbols, cmd.sources, cmd.lookback_hours)

    elif isinstance(cmd, BacktestRequest):
        return run_backtest(
            cmd.strategy_name,
            cmd.symbols,
            cmd.start_date,
            cmd.end_date,
            cmd.initial_capital,
        )

    elif isinstance(cmd, ReportTaskCompletion):
        # Display final answer to user
        console.print("\n" + "=" * 80)
        console.print(
            "[bold green]🎯 ФИНАЛЬНЫЙ АНАЛИЗ / FINAL TRADING ANALYSIS[/bold green]"
        )
        console.print("=" * 80)

        # Show final recommendation
        if hasattr(cmd, "final_recommendation") and cmd.final_recommendation:
            console.print(
                Panel(
                    cmd.final_recommendation,
                    title="[bold yellow]📈 ТОРГОВАЯ РЕКОМЕНДАЦИЯ[/bold yellow]",
                    border_style="yellow",
                )
            )

        # Show completed steps
        console.print("\n[bold cyan]📋 ВЫПОЛНЕННЫЕ ШАГИ АНАЛИЗА:[/bold cyan]")
        for i, step in enumerate(cmd.completed_steps_laconic, 1):
            console.print(f"[blue]{i}.[/blue] {step}")

        # Show current portfolio status
        if DB.memory.positions:
            console.print("\n[bold cyan]💼 ТЕКУЩИЕ ПОЗИЦИИ:[/bold cyan]")
            for position in DB.memory.positions[-5:]:  # Show last 5 positions
                console.print(
                    f"• {position.symbol}: {position.quantity} shares @ ${position.avg_cost:.2f}"
                )

        console.print("=" * 80 + "\n")

        return {
            "completed_steps": cmd.completed_steps_laconic,
            "final_recommendation": getattr(cmd, "final_recommendation", ""),
            "status": cmd.code,
            "message": f"Financial analysis {cmd.code} completed with {len(cmd.completed_steps_laconic)} steps",
            "final_answer_displayed": True,
        }

    else:
        return f"Unknown tool: {type(cmd)}"


def get_market_data(
    symbols: List[str], timeframe: str = "1d", period: str = "1mo"
) -> Dict[str, Any]:
    """Get market data for specified symbols"""
    try:
        logger.info(f"Fetching market data for {symbols}")

        # Use market data collector
        market_response = market_collector.get_market_data(symbols, timeframe, period)

        # Store in database
        DB.add_market_data(market_response.market_data)

        # Convert to dict for JSON serialization
        result = {
            "symbols_analyzed": market_response.symbols_analyzed,
            "data_points": len(market_response.market_data),
            "market_trend": market_response.market_trend,
            "volatility_assessment": market_response.volatility_assessment,
            "key_insights": market_response.key_insights,
            "timestamp": market_response.timestamp.isoformat(),
            "market_data": [data.model_dump() for data in market_response.market_data],
        }

        logger.info(
            f"Successfully fetched data for {len(market_response.market_data)} symbols"
        )
        return result

    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return {"error": str(e), "symbols_requested": symbols, "success": False}


def analyze_trading_opportunity(
    symbol: str,
    analysis_type: str = "comprehensive",
    budget: float = None,
    risk_tolerance: str = "medium",
) -> Dict[str, Any]:
    """Analyze trading opportunity for a specific symbol"""
    try:
        logger.info(f"Analyzing trading opportunity for {symbol}")

        # Use trading analyzer
        recommendation = trading_analyzer.analyze_trading_opportunity(
            symbol, analysis_type, budget, risk_tolerance
        )

        # Store analysis in history
        analysis_record = {
            "symbol": symbol,
            "analysis_type": analysis_type,
            "recommendation": recommendation.model_dump(),
            "timestamp": datetime.now().isoformat(),
        }
        DB.analysis_history.append(analysis_record)

        # Convert to dict for JSON serialization
        result = {
            "symbol": symbol,
            "recommendation": recommendation.recommendation,
            "target_price": recommendation.target_price,
            "stop_loss": recommendation.stop_loss,
            "position_size": recommendation.position_size,
            "confidence_level": recommendation.confidence_level,
            "rationale": recommendation.rationale,
            "risk_level": recommendation.risk_assessment.risk_level,
            "volatility": recommendation.risk_assessment.volatility,
            "max_drawdown": recommendation.risk_assessment.max_drawdown,
            "recommendations": recommendation.risk_assessment.recommendations,
            "timestamp": recommendation.timestamp.isoformat(),
            "analysis_type": analysis_type,
        }

        logger.info(f"Analysis complete for {symbol}: {recommendation.recommendation}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "recommendation": "hold",
            "rationale": f"Analysis failed due to error: {str(e)}",
        }


def generate_forecast(
    question: str,
    symbols: List[str] = None,
    forecast_horizon: str = "1_week",
    agent_types: List[str] = None,
) -> Dict[str, Any]:
    """Generate probabilistic forecast for market question"""
    try:
        logger.info(f"Generating forecast: {question}")

        if agent_types is None:
            agent_types = ["bullish", "bearish", "technical"]

        if symbols is None:
            symbols = settings.default_symbols[:3]  # Use first 3 default symbols

        # Simulate different agent perspectives
        forecasts = []

        for agent_type in agent_types:
            # Get market data for context
            market_response = market_collector.get_market_data(symbols, period="1mo")

            if market_response.market_data:
                avg_trend = (
                    1
                    if market_response.market_trend == "bullish"
                    else -1
                    if market_response.market_trend == "bearish"
                    else 0
                )

                # Simulate agent bias
                if agent_type == "bullish":
                    base_prob = 0.6 + (avg_trend * 0.1)
                    rationale = f"Bullish outlook based on positive market trends and technical indicators for {', '.join(symbols)}"
                elif agent_type == "bearish":
                    base_prob = 0.4 - (avg_trend * 0.1)
                    rationale = f"Bearish perspective considering market risks and volatility factors for {', '.join(symbols)}"
                else:  # technical
                    base_prob = 0.5 + (avg_trend * 0.05)
                    rationale = f"Technical analysis of price patterns and momentum indicators for {', '.join(symbols)}"

                # Add some randomness to simulate real forecasting uncertainty
                probability = max(0.1, min(0.9, base_prob + np.random.normal(0, 0.1)))

                forecast = ForecastResult(
                    question=question,
                    prediction_probability=probability,
                    confidence_interval={"80%": 0.15, "95%": 0.25},
                    rationale=rationale,
                    forecast_horizon=forecast_horizon,
                    agent_type=agent_type,
                    timestamp=datetime.now(),
                )

                forecasts.append(forecast)
                DB.add_forecast(forecast)

        # Calculate consensus
        if forecasts:
            avg_probability = np.mean([f.prediction_probability for f in forecasts])
            disagreement = np.std([f.prediction_probability for f in forecasts])
            confidence = max(
                0.1, 1.0 - disagreement
            )  # Higher disagreement = lower confidence
        else:
            avg_probability = 0.5
            disagreement = 0.0
            confidence = 0.5

        result = {
            "question": question,
            "consensus_probability": avg_probability,
            "individual_forecasts": [f.model_dump() for f in forecasts],
            "disagreement_level": disagreement,
            "confidence_score": confidence,
            "forecast_horizon": forecast_horizon,
            "symbols_analyzed": symbols,
            "agent_types": agent_types,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Forecast generated: {avg_probability:.1%} probability")
        return result

    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return {
            "error": str(e),
            "question": question,
            "consensus_probability": 0.5,
            "confidence_score": 0.0,
        }


def assess_risk(
    symbol: str = None, trade_amount: float = None, assessment_type: str = "portfolio"
) -> Dict[str, Any]:
    """Assess risk for trade or portfolio"""
    try:
        logger.info(f"Assessing {assessment_type} risk for {symbol or 'portfolio'}")

        if assessment_type == "portfolio" and DB.memory.market_data:
            # Get unique symbols from recent market data
            symbols = list(set([data.symbol for data in DB.memory.market_data[-20:]]))
            amounts = [1.0] * len(symbols)  # Equal weights
        elif symbol:
            symbols = [symbol]
            amounts = [trade_amount or 1.0]
        else:
            symbols = settings.default_symbols[:3]
            amounts = [1.0, 1.0, 1.0]

        # Use risk analyzer
        risk_assessment = risk_analyzer.assess_portfolio_risk(symbols, amounts)

        result = {
            "assessment_type": assessment_type,
            "symbols_analyzed": symbols,
            "var_1d": risk_assessment.var_1d,
            "var_5d": risk_assessment.var_5d,
            "max_drawdown": risk_assessment.max_drawdown,
            "volatility": risk_assessment.volatility,
            "sharpe_ratio": risk_assessment.sharpe_ratio,
            "risk_level": risk_assessment.risk_level,
            "recommendations": risk_assessment.recommendations,
            "timestamp": risk_assessment.timestamp.isoformat(),
        }

        logger.info(
            f"Risk assessment complete: {risk_assessment.risk_level} risk level"
        )
        return result

    except Exception as e:
        logger.error(f"Error in risk assessment: {e}")
        return {
            "error": str(e),
            "assessment_type": assessment_type,
            "risk_level": "unknown",
            "recommendations": ["Risk assessment failed - manual review required"],
        }


def analyze_news(
    symbols: List[str] = None, sources: List[str] = None, lookback_hours: int = 24
) -> Dict[str, Any]:
    """Analyze news sentiment for symbols"""
    try:
        if symbols is None:
            symbols = settings.default_symbols[:3]

        logger.info(f"Analyzing news sentiment for {symbols}")

        # Get mock news sentiment (in real system would fetch from news APIs)
        news_sentiment = news_analyzer.get_mock_news_sentiment(symbols)

        # Calculate aggregate sentiment
        if news_sentiment:
            avg_sentiment = np.mean([news.sentiment_score for news in news_sentiment])
            sentiment_range = (
                min([news.sentiment_score for news in news_sentiment]),
                max([news.sentiment_score for news in news_sentiment]),
            )
        else:
            avg_sentiment = 0.0
            sentiment_range = (0.0, 0.0)

        # Categorize sentiment
        if avg_sentiment > 0.2:
            sentiment_category = "positive"
        elif avg_sentiment < -0.2:
            sentiment_category = "negative"
        else:
            sentiment_category = "neutral"

        result = {
            "symbols_analyzed": symbols,
            "news_articles": len(news_sentiment),
            "average_sentiment": avg_sentiment,
            "sentiment_range": sentiment_range,
            "sentiment_category": sentiment_category,
            "lookback_hours": lookback_hours,
            "detailed_sentiment": [news.model_dump() for news in news_sentiment],
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"News analysis complete: {sentiment_category} sentiment ({avg_sentiment:.2f})"
        )
        return result

    except Exception as e:
        logger.error(f"Error analyzing news: {e}")
        return {
            "error": str(e),
            "symbols_analyzed": symbols or [],
            "sentiment_category": "unknown",
            "average_sentiment": 0.0,
        }


def run_backtest(
    strategy_name: str,
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
) -> Dict[str, Any]:
    """Run simplified backtest simulation"""
    try:
        logger.info(f"Running backtest for {strategy_name} strategy")

        # Simplified backtest simulation
        # In real system would implement proper backtesting engine

        # Generate mock backtest results
        total_trades = np.random.randint(10, 50)
        win_rate = np.random.uniform(0.45, 0.65)  # 45-65% win rate
        avg_return_per_trade = np.random.uniform(-0.02, 0.04)  # -2% to 4% per trade

        total_return = avg_return_per_trade * total_trades
        final_capital = initial_capital * (1 + total_return)

        # Calculate other metrics
        sharpe_ratio = np.random.uniform(0.5, 2.5)
        max_drawdown = np.random.uniform(0.05, 0.20)
        volatility = np.random.uniform(0.10, 0.30)

        result = {
            "strategy_name": strategy_name,
            "symbols": symbols,
            "period": f"{start_date} to {end_date}",
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_return_per_trade": avg_return_per_trade,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "performance_summary": f"Strategy returned {total_return:.1%} over the test period",
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Backtest complete: {total_return:.1%} return, {win_rate:.1%} win rate"
        )
        return result

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return {
            "error": str(e),
            "strategy_name": strategy_name,
            "symbols": symbols,
            "result": "Backtest failed",
        }


# ============ Azure OpenAI Integration ============


def setup_azure_client() -> AzureOpenAI:
    """Setup Azure OpenAI client using settings"""
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


def run_sgr_step(
    client: AzureOpenAI, task: str, conversation_log: List[Dict]
) -> SGRTradingResponse:
    """Execute one SGR reasoning step for financial analysis"""
    deployment_name = settings.azure_openai_deployment_name

    # Prepare system prompt with available financial tools
    system_prompt = f"""You are a professional financial trading assistant using Schema-Guided Reasoning for market analysis.

Available financial analysis tools:
- get_market_data: Get real-time market data for stocks, ETFs, crypto (symbols, timeframe, period)
- analyze_trading_opportunity: Comprehensive trading analysis with technical indicators and recommendations
- generate_forecast: Create probabilistic forecasts using multiple agent perspectives (bullish, bearish, technical)
- assess_risk: Portfolio and trade risk assessment with VaR, volatility, and drawdown analysis
- analyze_news: News sentiment analysis for market-moving events
- run_backtest: Historical strategy validation and performance metrics
- report_completion: Provide final trading recommendations and analysis summary

Financial data capabilities:
• Real-time market data via Yahoo Finance API
• Technical indicators: RSI, MACD, Bollinger Bands, Moving Averages
• Risk metrics: Value at Risk (VaR), Sharpe ratio, Maximum Drawdown
• Sentiment analysis: News and social media sentiment scoring
• Multi-agent forecasting: Bullish, bearish, and technical perspectives

Current market memory: {len(DB.memory.market_data)} data points, {len(DB.memory.forecasts)} forecasts
Paper trading mode: {"Enabled" if settings.enable_paper_trading else "Disabled"}
Risk tolerance settings: Max position {settings.max_position_size:.1%}, Max drawdown {settings.max_daily_drawdown:.1%}

Your task: {task}

Instructions for financial analysis:
- Start with market data collection for relevant symbols
- Perform comprehensive technical and fundamental analysis
- Generate probabilistic forecasts with confidence intervals
- Always assess risk before recommending trades
- Consider news sentiment and market conditions
- Provide specific price targets, stop losses, and position sizes
- Use professional financial terminology and justify all recommendations
- For portfolio analysis, consider diversification and correlation
- Always specify risk level (low/medium/high) and confidence level
- When analysis is complete, use report_completion with detailed trading recommendations
- Include specific entry/exit strategies and risk management guidelines
- Target institutional-grade analysis with Sharpe ratio >2.0 when possible"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_log

    try:
        # Use structured output for financial analysis
        completion = client.beta.chat.completions.parse(
            model=deployment_name,
            response_format=SGRTradingResponse,
            messages=messages,
            max_completion_tokens=1000,
        )

        sgr_response = completion.choices[0].message.parsed
        logger.info(f"SGR Financial Analysis: {sgr_response.current_state}")
        return sgr_response

    except Exception as e:
        logger.error(f"Error in SGR financial analysis step: {e}")
        raise


# ============ Main Agent Loop ============


def run_financial_agent(task: str, max_steps: int = None) -> None:
    """Run the SGR financial trading agent"""
    if max_steps is None:
        max_steps = settings.max_sgr_steps

    console.print(
        Panel(
            f"[bold blue]🚀 SGR Financial Trading Agent[/bold blue]\n\n"
            f"Task: {task}\n"
            f"Paper Trading: {'✓ Enabled' if settings.enable_paper_trading else '✗ Disabled'}\n"
            f"Risk Settings: {settings.max_position_size:.1%} max position, {settings.max_daily_drawdown:.1%} max drawdown",
            expand=False,
            border_style="blue",
        )
    )

    # Validate Azure OpenAI configuration
    try:
        is_valid, missing_keys = validate_required_keys()
        if not is_valid:
            console.print(
                f"[red]Missing required API keys: {', '.join(missing_keys)}[/red]"
            )
            console.print(
                "Please check your .env file and ensure all required credentials are configured."
            )
            return

        client = setup_azure_client()
    except Exception as e:
        console.print(f"[red]Error setting up Azure OpenAI client: {e}[/red]")
        return

    conversation_log = []

    for step in range(1, max_steps + 1):
        console.print(f"\n[bold yellow]📊 Analysis Step {step}[/bold yellow]")

        # Force completion after too many steps
        if step >= 8:
            console.print(
                f"[yellow]⚠ Forcing analysis completion after {step - 1} steps[/yellow]"
            )

            # Generate summary from conversation
            final_recommendation = generate_financial_summary(task, conversation_log)

            console.print("\n" + "=" * 80)
            console.print(
                "[bold green]🎯 ФИНАЛЬНЫЙ АНАЛИЗ / FINAL ANALYSIS[/bold green]"
            )
            console.print("=" * 80)
            console.print(
                Panel(
                    final_recommendation,
                    title="[bold yellow]📈 ТОРГОВАЯ РЕКОМЕНДАЦИЯ[/bold yellow]",
                    border_style="yellow",
                )
            )
            console.print("=" * 80 + "\n")
            break

        try:
            # Get SGR response
            sgr_response = run_sgr_step(client, task, conversation_log)

            # Show current analysis state
            console.print(
                f"[cyan]Current analysis:[/cyan] {sgr_response.current_state}"
            )

            # Show planned steps
            if sgr_response.plan_remaining_steps_brief:
                console.print("[cyan]Planned analysis steps:[/cyan]")
                for i, step_desc in enumerate(
                    sgr_response.plan_remaining_steps_brief, 1
                ):
                    console.print(f"  {i}. {step_desc}")

            # Check if analysis is completed
            if isinstance(sgr_response.function, ReportTaskCompletion):
                console.print(
                    f"[green]✓ Financial analysis {sgr_response.function.code}![/green]"
                )
                result = dispatch(sgr_response.function)
                result_text = json.dumps(result, ensure_ascii=False, default=str)
                conversation_log.append(
                    {
                        "role": "tool",
                        "content": result_text,
                        "tool_call_id": f"step_{step}",
                    }
                )
                break

            # Show selected tool
            console.print(f"[green]Executing:[/green] {sgr_response.function.tool}")

            # Create a table for tool parameters
            if hasattr(sgr_response.function, "model_dump"):
                params = sgr_response.function.model_dump()
                table = Table(
                    title="Tool Parameters",
                    show_header=True,
                    header_style="bold magenta",
                )
                table.add_column("Parameter", style="dim")
                table.add_column("Value")

                for key, value in params.items():
                    if key != "tool":  # Skip the tool name
                        table.add_row(key, str(value))
                console.print(table)

            # Add assistant response to log
            conversation_log.append(
                {
                    "role": "assistant",
                    "content": sgr_response.plan_remaining_steps_brief[0]
                    if sgr_response.plan_remaining_steps_brief
                    else "Processing...",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": f"step_{step}",
                            "function": {
                                "name": sgr_response.function.tool,
                                "arguments": sgr_response.function.model_dump_json(),
                            },
                        }
                    ],
                }
            )

            # Execute the tool
            result = dispatch(sgr_response.function)

            # Show result with nice formatting
            if isinstance(result, dict):
                # Format financial data nicely
                if "market_data" in result:
                    console.print(
                        f"[green]📈 Market Data Retrieved:[/green] {result.get('symbols_analyzed', [])} - {result.get('market_trend', 'unknown')} trend"
                    )
                elif "recommendation" in result:
                    rec = result.get("recommendation", "hold")
                    symbol = result.get("symbol", "")
                    confidence = result.get("confidence_level", 0) * 100
                    console.print(
                        f"[green]🎯 Trading Recommendation:[/green] {rec.upper()} {symbol} (confidence: {confidence:.0f}%)"
                    )
                elif "consensus_probability" in result:
                    prob = result.get("consensus_probability", 0.5) * 100
                    question = result.get("question", "")
                    console.print(
                        f"[green]🔮 Forecast:[/green] {prob:.0f}% probability - {question}"
                    )
                else:
                    # General result display
                    result_json = json.dumps(
                        result, indent=2, ensure_ascii=False, default=str
                    )
                    syntax = Syntax(
                        result_json, "json", theme="monokai", line_numbers=True
                    )
                    console.print(syntax)
            else:
                console.print(f"[yellow]Result:[/yellow] {result}")

            # Add tool result to conversation log
            result_text = (
                json.dumps(result, ensure_ascii=False, default=str)
                if isinstance(result, dict)
                else str(result)
            )
            conversation_log.append(
                {"role": "tool", "content": result_text, "tool_call_id": f"step_{step}"}
            )

            # Check if we should continue
            if sgr_response.task_completed:
                console.print("[green]✓ Financial analysis completed by agent![/green]")
                break

        except Exception as e:
            console.print(f"[red]Error in analysis step {step}: {e}[/red]")
            break

    # Show summary
    console.print(f"\n[bold]Financial analysis completed in {step} steps[/bold]")

    # Show performance summary if available
    if DB.memory.market_data:
        console.print(f"[dim]Data points collected: {len(DB.memory.market_data)}[/dim]")
    if DB.memory.forecasts:
        console.print(f"[dim]Forecasts generated: {len(DB.memory.forecasts)}[/dim]")


def generate_financial_summary(task: str, conversation_log: List[Dict]) -> str:
    """Generate a financial analysis summary from conversation history"""
    # Extract key financial metrics and recommendations
    summary_parts = [
        f"Анализ по запросу: {task}",
        "",
        "Ключевые результаты анализа:",
    ]

    # Look for market data, recommendations, and forecasts in conversation
    steps_completed = len(
        [entry for entry in conversation_log if entry.get("role") == "assistant"]
    )
    tools_used = []

    for entry in conversation_log:
        if entry.get("role") == "tool":
            content = entry.get("content", "")
            if "market_data" in content:
                tools_used.append("• Получены рыночные данные и технические индикаторы")
            elif "recommendation" in content:
                tools_used.append("• Выполнен анализ торговых возможностей")
            elif "consensus_probability" in content:
                tools_used.append("• Созданы вероятностные прогнозы")
            elif "risk_level" in content:
                tools_used.append("• Проведена оценка рисков портфеля")

    if tools_used:
        summary_parts.extend(tools_used)

    summary_parts.extend(
        [
            "",
            f"Всего выполнено шагов анализа: {steps_completed}",
            "",
            "Рекомендации:",
            "• Используйте только бумажную торговлю для тестирования стратегий",
            "• Не превышайте установленные лимиты риска",
            "• Регулярно пересматривайте позиции при изменении рыночных условий",
            "• Учитывайте макроэкономические факторы при принятии решений",
        ]
    )

    return "\n".join(summary_parts)


# ============ CLI Interface ============


def main():
    """Main CLI interface for financial trading agent"""
    console.print("[bold green]🚀 SGR Financial Trading Agent[/bold green]")
    console.print(
        "Multi-agent AI system for financial analysis and algorithmic trading"
    )
    console.print("Based on Schema-Guided Reasoning pattern")
    console.print()

    # Check configuration
    is_valid, missing_keys = validate_required_keys()
    if is_valid:
        console.print("[green]✓ Azure OpenAI configuration validated[/green]")
    else:
        console.print(
            f"[red]✗ Missing required configuration: {', '.join(missing_keys)}[/red]"
        )
        console.print("Please create a .env file with the required API keys.")
        return

    # Show available data sources
    console.print(
        f"[green]✓ Paper trading mode: {'Enabled' if settings.enable_paper_trading else 'Disabled'}[/green]"
    )
    console.print(
        f"[dim]Default symbols: {', '.join(settings.default_symbols[:5])}...[/dim]"
    )
    console.print(
        f"[dim]Risk limits: {settings.max_position_size:.1%} max position, {settings.max_daily_drawdown:.1%} max drawdown[/dim]"
    )

    # Example tasks from AGENTS.md
    example_tasks = [
        "Проанализируй текущую рыночную ситуацию по S&P 500",
        "Создай прогноз на следующую неделю для AAPL с оценкой риска",
        "Рекомендуй портфель акций с бюджетом $100,000 и толерантностью к риску 15%",
        "Проведи бэктест стратегии momentum на данных за последний год",
        "Оцени влияние решения ФРС по ставкам на технологический сектор",
    ]

    console.print("\n[bold]Example financial analysis tasks:[/bold]")
    for i, task in enumerate(example_tasks, 1):
        console.print(f"  {i}. {task}")

    while True:
        console.print()
        task_input = console.input(
            "[cyan]Enter your financial analysis task (or 'quit' to exit): [/cyan]"
        )

        if task_input.lower() in ["quit", "exit", "q"]:
            break

        if task_input.strip():
            # Check if user entered a number referring to example tasks
            if task_input.strip().isdigit():
                task_num = int(task_input.strip())
                if 1 <= task_num <= len(example_tasks):
                    task = example_tasks[task_num - 1]
                    console.print(f"[green]Selected example task:[/green] {task}")
                else:
                    console.print(
                        f"[yellow]Invalid task number. Please choose 1-{len(example_tasks)}[/yellow]"
                    )
                    continue
            else:
                task = task_input.strip()

            run_financial_agent(task)
        else:
            console.print("[yellow]Please enter a financial analysis task[/yellow]")

    console.print("[green]Happy trading! 📈[/green]")


if __name__ == "__main__":
    main()
