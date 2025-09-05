#!/usr/bin/env python3
"""
Market Data Tools for Financial Trading Agent

Tools for collecting and analyzing market data from various sources.
Includes Yahoo Finance integration, news sentiment, and basic technical analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from models import (
    MarketData,
    NewsSentiment,
    EconomicIndicator,
    ForecastResult,
    TradingSignal,
    RiskAssessment,
    MarketDataResponse,
    TradingRecommendation,
)
from settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""

    sma_20: float = 0.0
    sma_50: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    volatility: float = 0.0


class MarketDataCollector:
    """Market data collection and analysis"""

    def __init__(self):
        self.cache = {}  # Simple cache for recent data

    def get_market_data(
        self, symbols: List[str], timeframe: str = "1d", period: str = "1mo"
    ) -> MarketDataResponse:
        """
        Collect market data for given symbols

        Args:
            symbols: List of stock symbols (e.g., ["AAPL", "GOOGL"])
            timeframe: Data interval ("1m", "5m", "1h", "1d", "1w")
            period: Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
        """
        market_data_list = []
        key_insights = []

        try:
            for symbol in symbols:
                logger.info(f"Fetching data for {symbol}")

                # Get data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=timeframe)

                if hist.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue

                # Get the latest data point
                latest = hist.iloc[-1]

                # Calculate technical indicators
                tech_indicators = self._calculate_technical_indicators(hist)

                # Create MarketData object
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=float(latest["Open"]),
                    high=float(latest["High"]),
                    low=float(latest["Low"]),
                    close=float(latest["Close"]),
                    volume=int(latest["Volume"]),
                    volatility=tech_indicators.volatility,
                )

                market_data_list.append(market_data)

                # Generate insights
                price_change = (
                    (latest["Close"] - latest["Open"]) / latest["Open"]
                ) * 100
                if abs(price_change) > 2:
                    direction = "up" if price_change > 0 else "down"
                    key_insights.append(
                        f"{symbol}: Significant move {direction} ({price_change:.1f}%)"
                    )

                if tech_indicators.rsi > 70:
                    key_insights.append(
                        f"{symbol}: RSI indicates overbought condition ({tech_indicators.rsi:.1f})"
                    )
                elif tech_indicators.rsi < 30:
                    key_insights.append(
                        f"{symbol}: RSI indicates oversold condition ({tech_indicators.rsi:.1f})"
                    )

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            key_insights.append(f"Error fetching data: {str(e)}")

        # Determine overall market trend
        if market_data_list:
            avg_change = np.mean(
                [
                    ((data.close - data.open) / data.open) * 100
                    for data in market_data_list
                ]
            )

            if avg_change > 1:
                market_trend = "bullish"
            elif avg_change < -1:
                market_trend = "bearish"
            else:
                market_trend = "sideways"

            # Assess volatility
            avg_volatility = np.mean(
                [data.volatility or 0 for data in market_data_list]
            )
            if avg_volatility > 0.03:  # 3%
                volatility_assessment = "high"
            elif avg_volatility > 0.015:  # 1.5%
                volatility_assessment = "normal"
            else:
                volatility_assessment = "low"
        else:
            market_trend = "unknown"
            volatility_assessment = "unknown"

        return MarketDataResponse(
            symbols_analyzed=symbols,
            market_data=market_data_list,
            key_insights=key_insights,
            market_trend=market_trend,
            volatility_assessment=volatility_assessment,
            timestamp=datetime.now(),
        )

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate technical indicators from price data"""
        if len(df) < 20:
            return TechnicalIndicators()  # Return defaults if insufficient data

        try:
            # Simple Moving Averages
            sma_20 = (
                df["Close"].rolling(window=20).mean().iloc[-1]
                if len(df) >= 20
                else df["Close"].iloc[-1]
            )
            sma_50 = (
                df["Close"].rolling(window=50).mean().iloc[-1]
                if len(df) >= 50
                else df["Close"].iloc[-1]
            )

            # RSI calculation
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not rs.iloc[-1] == 0 else 50

            # MACD (simplified)
            ema_12 = df["Close"].ewm(span=12).mean()
            ema_26 = df["Close"].ewm(span=26).mean()
            macd = (ema_12 - ema_26).iloc[-1]

            # Bollinger Bands
            sma_20_full = df["Close"].rolling(window=20).mean()
            std_20 = df["Close"].rolling(window=20).std()
            bollinger_upper = (sma_20_full + (std_20 * 2)).iloc[-1]
            bollinger_lower = (sma_20_full - (std_20 * 2)).iloc[-1]

            # Volatility (20-day)
            volatility = df["Close"].pct_change().rolling(window=20).std().iloc[
                -1
            ] * np.sqrt(252)  # Annualized

            return TechnicalIndicators(
                sma_20=float(sma_20),
                sma_50=float(sma_50),
                rsi=float(rsi) if not np.isnan(rsi) else 50.0,
                macd=float(macd) if not np.isnan(macd) else 0.0,
                bollinger_upper=float(bollinger_upper)
                if not np.isnan(bollinger_upper)
                else 0.0,
                bollinger_lower=float(bollinger_lower)
                if not np.isnan(bollinger_lower)
                else 0.0,
                volatility=float(volatility) if not np.isnan(volatility) else 0.0,
            )

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return TechnicalIndicators()


class NewsAnalyzer:
    """News sentiment analysis (simplified)"""

    def __init__(self):
        # Simple keyword-based sentiment (in real system would use NLP models)
        self.positive_keywords = [
            "gain",
            "profit",
            "growth",
            "bullish",
            "rally",
            "surge",
            "breakthrough",
            "positive",
            "upgrade",
            "beat",
            "exceed",
            "strong",
            "robust",
        ]
        self.negative_keywords = [
            "loss",
            "decline",
            "bearish",
            "crash",
            "plunge",
            "negative",
            "downgrade",
            "miss",
            "weak",
            "concern",
            "risk",
            "volatility",
            "uncertainty",
        ]

    def analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis
        Returns score between -1 (very negative) and 1 (very positive)
        """
        text_lower = text.lower()

        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return 0.0

        # Calculate sentiment score
        sentiment = (positive_count - negative_count) / max(total_words / 10, 1)
        return max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]

    def get_mock_news_sentiment(self, symbols: List[str]) -> List[NewsSentiment]:
        """
        Mock news sentiment (in real system would fetch from news APIs)
        """
        mock_news = []

        for symbol in symbols:
            # Generate mock news based on symbol
            if symbol in ["AAPL", "GOOGL", "MSFT"]:
                sentiment_score = 0.3  # Slightly positive for tech stocks
                headline = f"{symbol} reports strong quarterly earnings growth"
                content = f"{symbol} stock continues to show positive momentum with strong quarterly results and positive analyst coverage."
            elif symbol in ["TSLA"]:
                sentiment_score = 0.1  # Neutral to slightly positive
                headline = (
                    f"{symbol} faces mixed analyst opinions on production targets"
                )
                content = f"{symbol} stock shows volatility as analysts debate production capacity and market expansion plans."
            else:
                sentiment_score = 0.0  # Neutral for others
                headline = f"{symbol} maintains stable trading amid market conditions"
                content = f"{symbol} stock shows stable performance in line with broader market trends."

            news_item = NewsSentiment(
                headline=headline,
                content=content,
                source="mock_financial_news",
                timestamp=datetime.now(),
                sentiment_score=sentiment_score,
                relevance_score=0.8,
                entities_mentioned=[symbol],
                impact_assessment="medium",
            )
            mock_news.append(news_item)

        return mock_news


class RiskAnalyzer:
    """Risk assessment and management"""

    def assess_portfolio_risk(
        self, symbols: List[str], amounts: List[float] = None
    ) -> RiskAssessment:
        """
        Assess portfolio risk based on symbols and position sizes
        """
        try:
            if amounts is None:
                amounts = [1.0] * len(symbols)  # Equal weights

            # Get historical data for risk calculation
            portfolio_returns = []
            total_amount = sum(amounts)

            for symbol, amount in zip(symbols, amounts):
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1y", interval="1d")

                    if not hist.empty:
                        returns = hist["Close"].pct_change().dropna()
                        weight = amount / total_amount
                        weighted_returns = returns * weight
                        portfolio_returns.append(weighted_returns)

                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {e}")
                    continue

            if not portfolio_returns:
                # Fallback risk assessment
                return RiskAssessment(
                    var_1d=0.02,
                    var_5d=0.05,
                    max_drawdown=0.10,
                    volatility=0.15,
                    risk_level="medium",
                    recommendations=["Insufficient data for accurate risk assessment"],
                    timestamp=datetime.now(),
                )

            # Combine portfolio returns
            combined_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)

            # Calculate risk metrics
            volatility = combined_returns.std() * np.sqrt(252)  # Annualized volatility
            var_1d = np.percentile(combined_returns, 5)  # 95% VaR (1-day)
            var_5d = np.sqrt(5) * var_1d  # 5-day VaR (simplified)

            # Calculate max drawdown
            cumulative_returns = (1 + combined_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())

            # Sharpe ratio (assuming 2% risk-free rate)
            avg_return = combined_returns.mean() * 252
            sharpe_ratio = (avg_return - 0.02) / volatility if volatility > 0 else 0

            # Determine risk level
            if volatility > 0.25 or max_drawdown > 0.15:
                risk_level = "high"
            elif volatility > 0.15 or max_drawdown > 0.10:
                risk_level = "medium"
            else:
                risk_level = "low"

            # Generate recommendations
            recommendations = []
            if max_drawdown > settings.max_drawdown_threshold:
                recommendations.append(
                    f"Max drawdown ({max_drawdown:.1%}) exceeds threshold"
                )
            if volatility > 0.20:
                recommendations.append("Consider diversification to reduce volatility")
            if sharpe_ratio < 1.0:
                recommendations.append("Risk-adjusted returns below target")
            if not recommendations:
                recommendations.append("Risk profile within acceptable parameters")

            return RiskAssessment(
                var_1d=abs(var_1d),
                var_5d=abs(var_5d),
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                risk_level=risk_level,
                recommendations=recommendations,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return RiskAssessment(
                var_1d=0.02,
                var_5d=0.05,
                max_drawdown=0.10,
                volatility=0.15,
                risk_level="unknown",
                recommendations=[f"Risk assessment error: {str(e)}"],
                timestamp=datetime.now(),
            )


class TradingAnalyzer:
    """Trading analysis and recommendations"""

    def __init__(self):
        self.market_collector = MarketDataCollector()
        self.risk_analyzer = RiskAnalyzer()
        self.news_analyzer = NewsAnalyzer()

    def analyze_trading_opportunity(
        self,
        symbol: str,
        analysis_type: str = "comprehensive",
        budget: Optional[float] = None,
        risk_tolerance: str = "medium",
    ) -> TradingRecommendation:
        """
        Analyze trading opportunity for a specific symbol
        """
        try:
            # Get market data
            market_response = self.market_collector.get_market_data(
                [symbol], period="3mo"
            )

            if not market_response.market_data:
                raise ValueError(f"No market data available for {symbol}")

            market_data = market_response.market_data[0]

            # Get historical data for analysis
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo", interval="1d")

            if hist.empty:
                raise ValueError(f"No historical data available for {symbol}")

            # Calculate technical indicators
            tech_indicators = self.market_collector._calculate_technical_indicators(
                hist
            )

            # Get news sentiment
            news_sentiment = self.news_analyzer.get_mock_news_sentiment([symbol])
            avg_sentiment = np.mean([news.sentiment_score for news in news_sentiment])

            # Generate trading signals
            signals = []

            # Technical analysis signals
            if market_data.close > tech_indicators.sma_20:
                signals.append("Price above 20-day SMA (bullish)")
            else:
                signals.append("Price below 20-day SMA (bearish)")

            if tech_indicators.rsi < 30:
                signals.append("RSI oversold (potential buy)")
            elif tech_indicators.rsi > 70:
                signals.append("RSI overbought (potential sell)")

            # Sentiment signal
            if avg_sentiment > 0.2:
                signals.append("Positive news sentiment")
            elif avg_sentiment < -0.2:
                signals.append("Negative news sentiment")

            # Determine recommendation
            bullish_signals = sum(
                1
                for s in signals
                if any(word in s.lower() for word in ["bullish", "buy", "positive"])
            )
            bearish_signals = sum(
                1
                for s in signals
                if any(word in s.lower() for word in ["bearish", "sell", "negative"])
            )

            if bullish_signals > bearish_signals + 1:
                recommendation = "buy"
                position_size = (
                    0.05
                    if risk_tolerance == "low"
                    else 0.10
                    if risk_tolerance == "medium"
                    else 0.15
                )
            elif bearish_signals > bullish_signals + 1:
                recommendation = "sell"
                position_size = 0.0
            else:
                recommendation = "hold"
                position_size = 0.02  # Small position

            # Set target price and stop loss (simplified)
            current_price = market_data.close
            if recommendation == "buy":
                target_price = current_price * 1.10  # 10% upside target
                stop_loss = current_price * 0.95  # 5% stop loss
            elif recommendation == "sell":
                target_price = current_price * 0.90  # 10% downside target
                stop_loss = current_price * 1.05  # 5% stop loss
            else:
                target_price = current_price
                stop_loss = current_price * 0.97  # Tight stop for hold

            # Risk assessment for this position
            risk_assessment = self.risk_analyzer.assess_portfolio_risk(
                [symbol], [position_size]
            )

            # Generate rationale
            rationale = f"Technical Analysis: {'; '.join(signals)}. "
            rationale += f"News Sentiment: {avg_sentiment:.2f} (range: -1 to 1). "
            rationale += f"Current price: ${current_price:.2f}, Volatility: {tech_indicators.volatility:.1%}."

            # Confidence level based on signal strength and data quality
            confidence_factors = [
                min(
                    abs(bullish_signals - bearish_signals) / max(len(signals), 1), 1.0
                ),  # Signal clarity
                min(abs(avg_sentiment), 1.0),  # Sentiment strength
                min(
                    len(hist) / 60, 1.0
                ),  # Data sufficiency (60 days = full confidence)
            ]
            confidence_level = np.mean(confidence_factors)

            return TradingRecommendation(
                symbol=symbol,
                recommendation=recommendation,
                target_price=target_price,
                stop_loss=stop_loss,
                position_size=position_size,
                rationale=rationale,
                risk_assessment=risk_assessment,
                confidence_level=confidence_level,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error analyzing trading opportunity for {symbol}: {e}")

            # Return conservative fallback recommendation
            return TradingRecommendation(
                symbol=symbol,
                recommendation="hold",
                target_price=None,
                stop_loss=None,
                position_size=0.0,
                rationale=f"Analysis error: {str(e)}. Recommending hold until manual review.",
                risk_assessment=RiskAssessment(
                    var_1d=0.02,
                    var_5d=0.05,
                    max_drawdown=0.10,
                    volatility=0.15,
                    risk_level="unknown",
                    recommendations=["Manual review required"],
                    timestamp=datetime.now(),
                ),
                confidence_level=0.0,
                timestamp=datetime.now(),
            )
