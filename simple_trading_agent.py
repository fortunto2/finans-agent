#!/usr/bin/env python3
"""
Simple SGR Financial Trading Agent

Simplified SGR agent with Azure OpenAI for financial analysis.
Uses only Yahoo Finance data, no external news APIs.
"""

import json
import uuid
from typing import List, Union, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import AzureOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
import logging
from datetime import datetime
import yfinance as yf
import numpy as np

from settings import settings, validate_required_keys
from api import OpointAPI

# Setup console and logging
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ Simple Models for SGR ============


class SimpleMarketDataRequest(BaseModel):
    """Get market data for symbols"""

    tool: Literal["get_market_data"]
    symbols: List[str]
    period: str = "1mo"

    class Config:
        extra = "forbid"


class SimpleForecastRequest(BaseModel):
    """Generate forecast for symbol"""

    tool: Literal["generate_forecast"]
    symbol: str
    question: str
    timeframe: str = "1_week"

    class Config:
        extra = "forbid"


class SimpleAnalysisRequest(BaseModel):
    """Analyze trading opportunity"""

    tool: Literal["analyze_trading"]
    symbol: str
    budget: Optional[float] = 10000
    risk_tolerance: str = "medium"

    class Config:
        extra = "forbid"


class SimpleRiskRequest(BaseModel):
    """Assess portfolio risk"""

    tool: Literal["assess_risk"]
    symbols: List[str]
    amounts: Optional[List[float]] = None

    class Config:
        extra = "forbid"


class SimpleNewsRequest(BaseModel):
    """Analyze news sentiment for symbols"""

    tool: Literal["analyze_news"]
    symbols: List[str]
    search_text: Optional[str] = None
    num_articles: int = 10

    class Config:
        extra = "forbid"


class SimpleCompletionRequest(BaseModel):
    """Complete analysis with summary"""

    tool: Literal["complete_analysis"]
    summary: str
    recommendation: str
    confidence: float

    class Config:
        extra = "forbid"


class SimpleSGRResponse(BaseModel):
    """SGR Response with structured reasoning"""

    current_state: str = Field(..., description="Current market analysis state")
    plan_remaining_steps_brief: List[str] = Field(
        ..., description="List of remaining analysis steps"
    )
    task_completed: bool = Field(False, description="Whether analysis is completed")
    function: Union[
        SimpleMarketDataRequest,
        SimpleForecastRequest,
        SimpleAnalysisRequest,
        SimpleRiskRequest,
        SimpleNewsRequest,
        SimpleCompletionRequest,
    ] = Field(..., description="Next function to execute")

    class Config:
        extra = "forbid"


# ============ Simple Tool Functions ============


def get_market_data(symbols: List[str], period: str = "1mo") -> Dict[str, Any]:
    """Get market data from Yahoo Finance"""
    try:
        logger.info(f"Fetching market data for {symbols}")

        market_data = []
        insights = []

        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                continue

            latest = hist.iloc[-1]

            # Calculate simple metrics
            price_change = ((latest["Close"] - latest["Open"]) / latest["Open"]) * 100
            avg_volume = hist["Volume"].mean()
            volatility = hist["Close"].pct_change().std() * np.sqrt(252)

            data = {
                "symbol": symbol,
                "current_price": float(latest["Close"]),
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "volume": int(latest["Volume"]),
                "price_change_pct": price_change,
                "volatility": volatility,
                "avg_volume": int(avg_volume),
            }

            market_data.append(data)

            # Generate insights
            if abs(price_change) > 2:
                direction = "up" if price_change > 0 else "down"
                insights.append(
                    f"{symbol}: Significant move {direction} ({price_change:.1f}%)"
                )

            if volatility > 0.3:
                insights.append(
                    f"{symbol}: High volatility detected ({volatility:.1%})"
                )

        # Determine overall trend
        if market_data:
            avg_change = np.mean([d["price_change_pct"] for d in market_data])
            if avg_change > 1:
                trend = "bullish"
            elif avg_change < -1:
                trend = "bearish"
            else:
                trend = "sideways"
        else:
            trend = "unknown"

        return {
            "symbols_analyzed": symbols,
            "market_data": market_data,
            "market_trend": trend,
            "key_insights": insights,
            "timestamp": datetime.now().isoformat(),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return {"error": str(e), "symbols": symbols, "success": False}


def generate_forecast(
    symbol: str, question: str, timeframe: str = "1_week"
) -> Dict[str, Any]:
    """Generate simple forecast based on technical analysis"""
    try:
        logger.info(f"Generating forecast for {symbol}")

        # Get historical data for analysis
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")

        if hist.empty:
            return {"error": f"No data for {symbol}", "symbol": symbol}

        # Calculate technical indicators
        current_price = hist["Close"].iloc[-1]
        sma_20 = hist["Close"].rolling(20).mean().iloc[-1]
        sma_50 = (
            hist["Close"].rolling(50).mean().iloc[-1]
            if len(hist) >= 50
            else current_price
        )

        # Calculate RSI
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if not rs.iloc[-1] == 0 else 50

        # Generate forecast based on technicals
        signals = []
        score = 0.5  # Neutral starting point

        if current_price > sma_20:
            signals.append("Price above 20-day moving average (bullish)")
            score += 0.1
        else:
            signals.append("Price below 20-day moving average (bearish)")
            score -= 0.1

        if current_price > sma_50:
            signals.append("Price above 50-day moving average (bullish)")
            score += 0.1
        else:
            signals.append("Price below 50-day moving average (bearish)")
            score -= 0.1

        if rsi < 30:
            signals.append("RSI oversold - potential upside")
            score += 0.15
        elif rsi > 70:
            signals.append("RSI overbought - potential downside")
            score -= 0.15

        # Volatility adjustment
        volatility = hist["Close"].pct_change().std()
        if volatility > 0.03:
            signals.append("High volatility - increased uncertainty")
            score = 0.5 + (score - 0.5) * 0.7  # Reduce confidence

        # Clamp probability
        probability = max(0.1, min(0.9, score))

        # Generate rationale
        rationale = f"Technical analysis for {symbol}: {', '.join(signals)}. Current price: ${current_price:.2f}, RSI: {rsi:.1f}"

        return {
            "symbol": symbol,
            "question": question,
            "probability": probability,
            "timeframe": timeframe,
            "confidence": abs(probability - 0.5) * 2,  # Distance from neutral
            "rationale": rationale,
            "technical_signals": signals,
            "current_price": current_price,
            "rsi": rsi,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return {"error": str(e), "symbol": symbol, "probability": 0.5}


def analyze_trading(
    symbol: str, budget: float = 10000, risk_tolerance: str = "medium"
) -> Dict[str, Any]:
    """Analyze trading opportunity"""
    try:
        logger.info(f"Analyzing trading opportunity for {symbol}")

        # Get market data
        market_result = get_market_data([symbol], "3mo")
        if not market_result.get("success"):
            return {"error": "Failed to get market data", "symbol": symbol}

        data = market_result["market_data"][0]
        current_price = data["current_price"]
        volatility = data["volatility"]

        # Simple recommendation logic
        if data["price_change_pct"] > 2 and volatility < 0.25:
            recommendation = "buy"
            confidence = 0.75
        elif data["price_change_pct"] < -2 and volatility < 0.25:
            recommendation = "sell"
            confidence = 0.70
        else:
            recommendation = "hold"
            confidence = 0.60

        # Position sizing based on risk tolerance
        if risk_tolerance == "low":
            position_size = 0.05  # 5%
        elif risk_tolerance == "high":
            position_size = 0.15  # 15%
        else:
            position_size = 0.10  # 10%

        # Calculate targets
        if recommendation == "buy":
            target_price = current_price * 1.10
            stop_loss = current_price * 0.95
        elif recommendation == "sell":
            target_price = current_price * 0.90
            stop_loss = current_price * 1.05
        else:
            target_price = current_price
            stop_loss = current_price * 0.97

        # Risk assessment
        risk_level = (
            "low" if volatility < 0.15 else "high" if volatility > 0.25 else "medium"
        )

        return {
            "symbol": symbol,
            "recommendation": recommendation,
            "confidence": confidence,
            "position_size": position_size,
            "current_price": current_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "risk_level": risk_level,
            "volatility": volatility,
            "rationale": f"Analysis based on price momentum ({data['price_change_pct']:.1f}%) and volatility ({volatility:.1%})",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error analyzing trading: {e}")
        return {"error": str(e), "symbol": symbol, "recommendation": "hold"}


def assess_risk(symbols: List[str], amounts: List[float] = None) -> Dict[str, Any]:
    """Simple portfolio risk assessment"""
    try:
        logger.info(f"Assessing risk for portfolio: {symbols}")

        if amounts is None:
            amounts = [1.0] * len(symbols)

        portfolio_data = []
        total_volatility = 0

        for symbol, amount in zip(symbols, amounts):
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if not hist.empty:
                returns = hist["Close"].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)

                portfolio_data.append(
                    {
                        "symbol": symbol,
                        "weight": amount / sum(amounts),
                        "volatility": volatility,
                    }
                )

                total_volatility += volatility * (amount / sum(amounts))

        # Simple risk metrics
        if total_volatility > 0.25:
            risk_level = "high"
        elif total_volatility > 0.15:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Risk recommendations
        recommendations = []
        if total_volatility > 0.20:
            recommendations.append("Consider diversification to reduce volatility")
        if len(symbols) < 3:
            recommendations.append(
                "Portfolio may benefit from additional diversification"
            )
        if not recommendations:
            recommendations.append("Risk profile appears acceptable")

        return {
            "symbols": symbols,
            "portfolio_volatility": total_volatility,
            "risk_level": risk_level,
            "var_1d": total_volatility / np.sqrt(252),  # Simplified 1-day VaR
            "recommendations": recommendations,
            "portfolio_data": portfolio_data,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error assessing risk: {e}")
        return {"error": str(e), "symbols": symbols, "risk_level": "unknown"}


def analyze_news_sentiment(
    symbols: List[str], search_text: Optional[str] = None, num_articles: int = 10
) -> Dict[str, Any]:
    """Analyze news sentiment using Opoint API"""
    try:
        logger.info(f"Analyzing news sentiment for {symbols}")

        if not settings.opoint_api_key:
            logger.warning("Opoint API key not configured, using mock data")
            return _get_mock_news_sentiment(symbols)

        # Initialize Opoint API
        opoint = OpointAPI(settings.opoint_api_key)

        all_news_data = []
        overall_sentiment_scores = []

        for symbol in symbols:
            # Search for news about this symbol
            symbol_search = f"{symbol} OR {_get_company_name(symbol)}"
            if search_text:
                symbol_search = f"({symbol_search}) AND ({search_text})"

            # Get articles from last 7 days
            from datetime import timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            articles_df = opoint.search_site_and_articles(
                search_text=symbol_search,
                language="en",
                num_articles=num_articles,
                start_date=start_date,
                end_date=end_date,
            )

            if articles_df.empty:
                logger.warning(f"No news found for {symbol}")
                continue

            # Analyze sentiment of articles
            symbol_sentiment = 0.0
            symbol_articles = []

            for _, article in articles_df.iterrows():
                # Simple sentiment analysis based on keywords
                title = str(article.get("title", "")).lower()
                summary = str(article.get("summary", "")).lower()
                text = title + " " + summary

                # Advanced sentiment scoring (like in the document)
                sentiment_score = _analyze_article_sentiment(title, summary, symbol)
                
                # Causality analysis - check for market-moving events
                market_impact = _assess_market_impact(title, summary, symbol)
                
                # News importance scoring
                importance_score = _calculate_news_importance(article, symbol)

                symbol_sentiment += sentiment_score

                symbol_articles.append(
                    {
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "published_date": str(article.get("published_date", "")),
                        "source_name": article.get("source_name", ""),
                        "sentiment_score": sentiment_score,
                        "market_impact": market_impact,
                        "importance_score": importance_score,
                        "weighted_sentiment": sentiment_score * importance_score  # More credible sources have higher weight
                    }
                )

            # Average sentiment for this symbol
            if len(symbol_articles) > 0:
                avg_sentiment = symbol_sentiment / len(symbol_articles)
            else:
                avg_sentiment = 0.0

            overall_sentiment_scores.append(avg_sentiment)

            all_news_data.append(
                {
                    "symbol": symbol,
                    "sentiment_score": avg_sentiment,
                    "article_count": len(symbol_articles),
                    "articles": symbol_articles[:5],  # Top 5 articles
                }
            )

        # Calculate overall sentiment
        if overall_sentiment_scores:
            overall_sentiment = np.mean(overall_sentiment_scores)
            sentiment_label = _get_sentiment_label(overall_sentiment)
        else:
            overall_sentiment = 0.0
            sentiment_label = "neutral"

        return {
            "symbols": symbols,
            "overall_sentiment": overall_sentiment,
            "sentiment_label": sentiment_label,
            "news_data": all_news_data,
            "total_articles": sum(len(data["articles"]) for data in all_news_data),
            "timestamp": datetime.now().isoformat(),
            "data_source": "opoint",
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error analyzing news sentiment: {e}")
        # Fallback to mock data
        return _get_mock_news_sentiment(symbols)


def _get_company_name(symbol: str) -> str:
    """Get company name for better news search"""
    company_names = {
        "AAPL": "Apple Inc",
        "GOOGL": "Google Alphabet",
        "MSFT": "Microsoft",
        "TSLA": "Tesla",
        "AMZN": "Amazon",
        "NVDA": "Nvidia",
        "META": "Meta Facebook",
        "NFLX": "Netflix",
    }
    return company_names.get(symbol, symbol)


def _get_sentiment_label(score: float) -> str:
    """Convert sentiment score to label"""
    if score > 0.2:
        return "very_positive"
    elif score > 0.05:
        return "positive"
    elif score < -0.2:
        return "very_negative"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"


def _analyze_article_sentiment(title: str, summary: str, symbol: str) -> float:
    """Advanced sentiment analysis as proposed in the trading document"""
    text = title + " " + summary
    
    # Enhanced sentiment keywords (financial-specific)
    sentiment_indicators = {
        # Very positive (0.8-1.0)
        "breakthrough": 0.9, "record": 0.8, "soars": 0.9, "surge": 0.8,
        "outperforms": 0.7, "beats expectations": 0.9, "all-time high": 1.0,
        
        # Positive (0.3-0.7)  
        "buy": 0.6, "bull": 0.7, "growth": 0.5, "up": 0.4, "rise": 0.5,
        "gain": 0.6, "profit": 0.7, "strong": 0.6, "beat": 0.7, "exceed": 0.6,
        "upgrade": 0.7, "optimistic": 0.6, "bullish": 0.8, "rally": 0.7,
        
        # Negative (-0.3 to -0.7)
        "sell": -0.6, "bear": -0.7, "down": -0.4, "fall": -0.5, "loss": -0.6,
        "weak": -0.5, "miss": -0.7, "decline": -0.5, "cut": -0.6, "warning": -0.6,
        "downgrade": -0.7, "pessimistic": -0.6, "bearish": -0.8, "plunge": -0.8,
        
        # Very negative (-0.8 to -1.0)
        "crash": -0.9, "collapse": -1.0, "crisis": -0.8, "disaster": -0.9,
        "scandal": -0.8, "investigation": -0.7, "lawsuit": -0.6, "fraud": -1.0
    }
    
    # Calculate weighted sentiment
    total_score = 0.0
    word_count = 0
    
    for phrase, weight in sentiment_indicators.items():
        if phrase in text:
            total_score += weight
            word_count += 1
    
    # Normalize by word count to avoid bias toward longer articles
    if word_count > 0:
        sentiment_score = total_score / word_count
    else:
        sentiment_score = 0.0
    
    # Company-specific adjustments (companies with different volatility)
    volatility_multipliers = {
        "TSLA": 1.2,  # Tesla news tends to be more impactful
        "NVDA": 1.1,  # AI/crypto exposure
        "AAPL": 0.9,  # More stable, less reactive
        "MSFT": 0.9,  # Enterprise stability
    }
    
    multiplier = volatility_multipliers.get(symbol, 1.0)
    return max(-1.0, min(1.0, sentiment_score * multiplier))


def _assess_market_impact(title: str, summary: str, symbol: str) -> float:
    """Assess potential market impact of news (causality analysis)"""
    text = title + " " + summary
    
    # Market-moving event indicators
    high_impact_events = [
        "earnings", "revenue", "guidance", "acquisition", "merger", "ipo",
        "fda approval", "partnership", "contract", "lawsuit", "regulatory",
        "ceo", "layoffs", "restructuring", "dividend", "stock split",
        "buyback", "bankruptcy", "delisting", "investigation"
    ]
    
    medium_impact_events = [
        "analyst", "rating", "price target", "recommendation", "conference",
        "product launch", "expansion", "hiring", "investment", "funding"
    ]
    
    # Score impact potential
    impact_score = 0.0
    
    for event in high_impact_events:
        if event in text:
            impact_score += 0.8
    
    for event in medium_impact_events:
        if event in text:
            impact_score += 0.4
    
    # Time sensitivity (recent events have higher impact)
    return min(1.0, impact_score)


def _calculate_news_importance(article: dict, symbol: str) -> float:
    """Calculate news importance score based on source and content quality"""
    
    # Source credibility weights
    source_weights = {
        "reuters": 1.0,
        "bloomberg": 1.0, 
        "wall street journal": 0.95,
        "financial times": 0.95,
        "cnbc": 0.8,
        "marketwatch": 0.7,
        "seeking alpha": 0.6,
        "yahoo finance": 0.5
    }
    
    source_name = str(article.get("source_name", "")).lower()
    source_weight = 0.3  # Default for unknown sources
    
    for source, weight in source_weights.items():
        if source in source_name:
            source_weight = weight
            break
    
    # Content quality indicators
    title = str(article.get("title", ""))
    summary = str(article.get("summary", ""))
    
    quality_score = 0.5  # Base score
    
    # Longer, more detailed articles tend to be more important
    if len(summary) > 200:
        quality_score += 0.2
    elif len(summary) > 100:
        quality_score += 0.1
    
    # Articles with specific numbers/data are more credible
    if any(char.isdigit() for char in title + summary):
        quality_score += 0.1
    
    # Articles mentioning the symbol directly are more relevant
    if symbol.upper() in title.upper():
        quality_score += 0.2
    
    return min(1.0, source_weight * quality_score)


def _get_mock_news_sentiment(symbols: List[str]) -> Dict[str, Any]:
    """Fallback mock news sentiment data"""
    mock_data = []

    for symbol in symbols:
        # Generate realistic sentiment based on symbol
        if symbol in ["AAPL", "MSFT", "GOOGL"]:
            sentiment = np.random.uniform(0.1, 0.3)  # Generally positive tech
        elif symbol in ["TSLA"]:
            sentiment = np.random.uniform(-0.1, 0.4)  # Volatile
        else:
            sentiment = np.random.uniform(-0.15, 0.15)  # Neutral

        mock_data.append(
            {
                "symbol": symbol,
                "sentiment_score": sentiment,
                "article_count": np.random.randint(3, 12),
                "articles": [
                    {
                        "title": f"Market Analysis: {symbol} Shows {'Positive' if sentiment > 0 else 'Mixed'} Signals",
                        "summary": f"Recent analysis suggests {symbol} {'continues strong performance' if sentiment > 0 else 'faces market headwinds'}.",
                        "url": f"https://example.com/news/{symbol.lower()}",
                        "published_date": datetime.now().strftime("%Y-%m-%d"),
                        "source_name": "Mock Financial News",
                        "sentiment_score": sentiment,
                    }
                ],
            }
        )

    overall_sentiment = np.mean([d["sentiment_score"] for d in mock_data])

    return {
        "symbols": symbols,
        "overall_sentiment": overall_sentiment,
        "sentiment_label": _get_sentiment_label(overall_sentiment),
        "news_data": mock_data,
        "total_articles": sum(d["article_count"] for d in mock_data),
        "timestamp": datetime.now().isoformat(),
        "data_source": "mock",
        "success": True,
    }


# ============ SGR Dispatch ============


def dispatch_simple(cmd) -> Dict[str, Any]:
    """Execute SGR commands"""

    if isinstance(cmd, SimpleMarketDataRequest):
        return get_market_data(cmd.symbols, cmd.period)

    elif isinstance(cmd, SimpleForecastRequest):
        return generate_forecast(cmd.symbol, cmd.question, cmd.timeframe)

    elif isinstance(cmd, SimpleAnalysisRequest):
        return analyze_trading(cmd.symbol, cmd.budget, cmd.risk_tolerance)

    elif isinstance(cmd, SimpleRiskRequest):
        return assess_risk(cmd.symbols, cmd.amounts)

    elif isinstance(cmd, SimpleNewsRequest):
        return analyze_news_sentiment(cmd.symbols, cmd.search_text, cmd.num_articles)

    elif isinstance(cmd, SimpleCompletionRequest):
        # Display final analysis
        console.print("\n" + "=" * 80)
        console.print("[bold green]🎯 ФИНАЛЬНЫЙ АНАЛИЗ / FINAL ANALYSIS[/bold green]")
        console.print("=" * 80)

        console.print(
            Panel(
                cmd.summary,
                title="[bold yellow]📈 ТОРГОВАЯ РЕКОМЕНДАЦИЯ[/bold yellow]",
                border_style="yellow",
            )
        )

        console.print(f"[cyan]Рекомендация:[/cyan] {cmd.recommendation}")
        console.print(f"[cyan]Уверенность:[/cyan] {cmd.confidence:.1%}")
        console.print("=" * 80 + "\n")

        return {
            "status": "completed",
            "summary": cmd.summary,
            "recommendation": cmd.recommendation,
            "confidence": cmd.confidence,
        }

    else:
        return {"error": f"Unknown command: {type(cmd)}"}


# ============ Azure OpenAI Integration ============


def setup_azure_client() -> AzureOpenAI:
    """Setup Azure OpenAI client"""
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


def run_sgr_step(
    client: AzureOpenAI, task: str, conversation_log: List[Dict]
) -> SimpleSGRResponse:
    """Execute one SGR reasoning step"""

    system_prompt = f"""You are a professional financial trading assistant using Schema-Guided Reasoning.

Available tools for financial analysis:
- get_market_data: Get real-time market data from Yahoo Finance (symbols, period)
- generate_forecast: Create probabilistic forecast for symbol (symbol, question, timeframe)  
- analyze_trading: Comprehensive trading analysis with recommendations (symbol, budget, risk_tolerance)
- assess_risk: Portfolio risk assessment (symbols, amounts)
- analyze_news: News sentiment analysis using Opoint API (symbols, search_text, num_articles)
- complete_analysis: Provide final summary and recommendation (summary, recommendation, confidence)

Current task: {task}

Paper trading mode: {"Enabled" if settings.enable_paper_trading else "Disabled"}
Risk limits: Max position {settings.max_position_size:.1%}, Max drawdown {settings.max_daily_drawdown:.1%}

Instructions:
- Start with market data collection for relevant symbols
- Perform technical analysis and generate forecasts
- Analyze trading opportunities with risk assessment
- Include news sentiment analysis for market context
- Provide specific recommendations with confidence levels
- Use complete_analysis when ready to give final recommendation
- Focus on technical indicators: RSI, moving averages, volatility
- Consider news sentiment and market narrative in analysis
- Consider risk tolerance in position sizing recommendations"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_log

    try:
        completion = client.beta.chat.completions.parse(
            model=settings.azure_openai_deployment_name,
            response_format=SimpleSGRResponse,
            messages=messages,
            max_completion_tokens=800,
        )

        return completion.choices[0].message.parsed

    except Exception as e:
        logger.error(f"Error in SGR step: {e}")
        raise


# ============ Main Agent Loop ============


def run_simple_trading_agent(task: str, max_steps: int = 8) -> None:
    """Run the simple SGR trading agent"""

    console.print(
        Panel(
            f"[bold blue]🚀 Simple SGR Trading Agent[/bold blue]\n\n"
            f"Task: {task}\n"
            f"Paper Trading: {'✓ Enabled' if settings.enable_paper_trading else '✗ Disabled'}\n"
            f"Max Steps: {max_steps}",
            expand=False,
            border_style="blue",
        )
    )

    # Validate configuration
    try:
        is_valid, missing_keys = validate_required_keys()
        if not is_valid:
            console.print(
                f"[red]Missing required API keys: {', '.join(missing_keys)}[/red]"
            )
            return

        client = setup_azure_client()
        console.print("[green]✓ Azure OpenAI client initialized[/green]")
    except Exception as e:
        console.print(f"[red]Error setting up Azure OpenAI: {e}[/red]")
        return

    conversation_log = []

    for step in range(1, max_steps + 1):
        console.print(f"\n[bold yellow]📊 Analysis Step {step}[/bold yellow]")

        try:
            # Get SGR response
            sgr_response = run_sgr_step(client, task, conversation_log)

            # Show current state
            console.print(f"[cyan]Current state:[/cyan] {sgr_response.current_state}")

            # Show planned steps
            if sgr_response.plan_remaining_steps_brief:
                console.print("[cyan]Planned steps:[/cyan]")
                for i, step_desc in enumerate(
                    sgr_response.plan_remaining_steps_brief, 1
                ):
                    console.print(f"  {i}. {step_desc}")

            # Show selected tool
            console.print(f"[green]Executing:[/green] {sgr_response.function.tool}")

            # Add assistant message to log
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
            result = dispatch_simple(sgr_response.function)

            # Show result with nice formatting
            if isinstance(result, dict) and "success" in result and result["success"]:
                if "market_data" in result:
                    symbols = result.get("symbols_analyzed", [])
                    trend = result.get("market_trend", "unknown")
                    console.print(
                        f"[green]📈 Market Data:[/green] {', '.join(symbols)} - {trend} trend"
                    )
                elif "recommendation" in result:
                    rec = result.get("recommendation", "hold")
                    conf = result.get("confidence", 0) * 100
                    console.print(
                        f"[green]🎯 Recommendation:[/green] {rec.upper()} ({conf:.0f}% confidence)"
                    )
                elif "probability" in result:
                    prob = result.get("probability", 0.5) * 100
                    console.print(
                        f"[green]🔮 Forecast:[/green] {prob:.0f}% probability"
                    )
                elif "sentiment_label" in result:
                    sentiment = result.get("sentiment_label", "neutral")
                    articles = result.get("total_articles", 0)
                    console.print(
                        f"[green]📰 News Sentiment:[/green] {sentiment.upper()} ({articles} articles)"
                    )
                else:
                    console.print(
                        f"[green]✓ Success:[/green] {result.get('status', 'completed')}"
                    )
            else:
                # Show detailed result for errors or complex data
                result_json = json.dumps(
                    result, indent=2, ensure_ascii=False, default=str
                )
                syntax = Syntax(
                    result_json, "json", theme="monokai", line_numbers=False
                )
                console.print(syntax)

            # Add tool result to conversation
            result_text = json.dumps(result, ensure_ascii=False, default=str)
            conversation_log.append(
                {"role": "tool", "content": result_text, "tool_call_id": f"step_{step}"}
            )

            # Check if completed
            if sgr_response.task_completed or isinstance(
                sgr_response.function, SimpleCompletionRequest
            ):
                console.print("[green]✅ Analysis completed![/green]")
                break

        except Exception as e:
            console.print(f"[red]Error in step {step}: {e}[/red]")
            break

    console.print(f"\n[bold]Simple SGR analysis completed in {step} steps[/bold]")


# ============ CLI Interface ============


def main():
    """Main CLI for simple trading agent"""
    console.print("[bold green]🚀 Simple SGR Financial Trading Agent[/bold green]")
    console.print("Simplified Schema-Guided Reasoning for financial analysis")
    console.print()

    # Example tasks
    example_tasks = [
        "Проанализируй AAPL для торговли на следующую неделю",
        "Создай прогноз для TSLA с оценкой риска",
        "Рекомендуй портфель из AAPL, GOOGL, MSFT с бюджетом $50,000",
        "Оцени риски инвестиций в технологический сектор",
    ]

    console.print("[bold]Example tasks:[/bold]")
    for i, task in enumerate(example_tasks, 1):
        console.print(f"  {i}. {task}")

    while True:
        console.print()
        task_input = console.input("[cyan]Enter your trading task (or 'quit'): [/cyan]")

        if task_input.lower() in ["quit", "exit", "q"]:
            break

        if task_input.strip():
            # Check if user entered a number
            if task_input.strip().isdigit():
                task_num = int(task_input.strip())
                if 1 <= task_num <= len(example_tasks):
                    task = example_tasks[task_num - 1]
                    console.print(f"[green]Selected:[/green] {task}")
                else:
                    console.print(
                        f"[yellow]Invalid number. Choose 1-{len(example_tasks)}[/yellow]"
                    )
                    continue
            else:
                task = task_input.strip()

            run_simple_trading_agent(task)
        else:
            console.print("[yellow]Please enter a task[/yellow]")

    console.print("[green]Happy trading! 📈[/green]")


if __name__ == "__main__":
    main()
