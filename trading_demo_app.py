#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Trading SGR Agent - Chainlit GUI Interface
Beautiful web interface for the financial trading agent with step-by-step visualization
"""

import json
from typing import Any, Dict, List, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import chainlit as cl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

from models import (
    SGRTradingResponse,
    MarketDataRequest,
    ForecastRequest,
    TradingAnalysisRequest,
    BacktestRequest,
    RiskAssessmentRequest,
    NewsAnalysisRequest,
    WebAnalysisRequest,
    ComprehensiveWebResearch,
    ReportTaskCompletion,
    # Alpha Factory models
    AlphaGenerationRequest,
    AlphaSpec,
    AlphaOp,
    # New enhanced system models
    CalibrationAnalysisRequest,
    RiskControlsRequest,
    EnhancedMetricsRequest,
)

from sgr_trading_agent import (
    get_market_data,
    analyze_trading_opportunity,
    generate_forecast,
    assess_risk,
    analyze_news,
    run_backtest,
    generate_alphas,  # Alpha Factory function
    analyze_calibration,  # Forecast calibration
    manage_risk_controls_wrapper,  # Risk controls
    calculate_enhanced_metrics_wrapper,  # Enhanced metrics
    create_chat_session,
    save_chat_message,
    get_chat_history,
    dispatch,
    DB,
)

# Import web intelligence tools
from web_intelligence import (
    analyze_web_content,
    research_financial_topic,
)

from settings import settings

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Initialize Azure OpenAI client using same pattern as sgr_trading_agent.py
client = AsyncAzureOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
)

# Trading cache for conversation history
TRADING_CACHE = {"analyses": [], "forecasts": [], "recommendations": []}

# Global session tracking
CURRENT_SESSION = {"session_id": None, "session_name": None}

# =============================================================================
# PLOTLY CHART CREATION FUNCTIONS
# =============================================================================


def create_candlestick_chart(data: Dict[str, Any], symbol: str = "STOCK") -> go.Figure:
    """Create an interactive candlestick chart with volume."""

    # Extract data or create sample data if not available
    if "price_data" in data and data["price_data"]:
        price_data = data["price_data"]
        df = pd.DataFrame(price_data)
    else:
        # Create sample data for demonstration
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        base_price = 100
        df = pd.DataFrame(
            {
                "date": dates,
                "open": np.random.normal(base_price, 2, 30),
                "high": np.random.normal(base_price + 2, 1.5, 30),
                "low": np.random.normal(base_price - 2, 1.5, 30),
                "close": np.random.normal(base_price, 2, 30),
                "volume": np.random.randint(1000000, 5000000, 30),
            }
        )
        # Ensure high >= max(open, close) and low <= min(open, close)
        df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
        df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))

    # Create subplots with secondary y-axis for volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"{symbol} Price", "Volume"),
        row_heights=[0.7, 0.3],
    )

    # Add candlestick chart
    candlestick = go.Candlestick(
        x=df["date"] if "date" in df.columns else df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=symbol,
        increasing_line_color="#00ff88",  # Green for gains
        decreasing_line_color="#ff4444",  # Red for losses
        increasing_fillcolor="#00ff88",
        decreasing_fillcolor="#ff4444",
    )

    fig.add_trace(candlestick, row=1, col=1)

    # Add volume bars if available
    if "volume" in df.columns:
        volume_colors = [
            "#00ff88" if close >= open else "#ff4444"
            for close, open in zip(df["close"], df["open"])
        ]

        volume_bars = go.Bar(
            x=df["date"] if "date" in df.columns else df.index,
            y=df["volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.6,
        )
        fig.add_trace(volume_bars, row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f"📈 {symbol} - Анализ цены и объема",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False,
        template="plotly_dark",
        font=dict(color="white"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Format axes
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text="Цена ($)", row=1, col=1)
    fig.update_yaxes(title_text="Объем", row=2, col=1)

    return fig


def create_forecast_probability_chart(forecast: Dict[str, Any]) -> go.Figure:
    """Create a probability visualization chart for forecasts."""

    individual_forecasts = forecast.get("individual_forecasts", [])
    consensus_probability = forecast.get("consensus_probability", 0.5) * 100

    if individual_forecasts:
        agents = [f.get("agent_type", "Unknown") for f in individual_forecasts]
        probabilities = [
            f.get("prediction_probability", 0.5) * 100 for f in individual_forecasts
        ]
    else:
        # Sample data
        agents = ["Bull Agent", "Bear Agent", "Technical Agent", "Sentiment Agent"]
        probabilities = [75, 25, 60, 45]

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Agent probability bars
    colors = ["#00ff88" if p > 50 else "#ff4444" for p in probabilities]

    fig.add_trace(
        go.Bar(
            x=agents,
            y=probabilities,
            name="Прогнозы агентов",
            marker_color=colors,
            text=[f"{p:.0f}%" for p in probabilities],
            textposition="outside",
        )
    )

    # Add consensus line
    fig.add_hline(
        y=consensus_probability,
        line_dash="dash",
        line_color="#ffaa00",
        annotation_text=f"Консенсус: {consensus_probability:.0f}%",
        annotation_position="top right",
    )

    # Add 50% reference line
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="gray",
        annotation_text="50% базовая линия",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="🔮 Прогнозные вероятности по агентам",
        xaxis_title="Агенты",
        yaxis_title="Вероятность (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_dark",
        height=400,
        showlegend=False,
        font=dict(color="white"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_risk_gauge_chart(risk: Dict[str, Any]) -> go.Figure:
    """Create a risk gauge chart with VaR and other metrics."""

    risk_level = risk.get("risk_level", "medium")
    var_1d = abs(risk.get("var_1d", 0.05)) * 100  # Convert to percentage
    var_5d = abs(risk.get("var_5d", 0.12)) * 100

    # Map risk level to numeric value
    risk_mapping = {"low": 25, "medium": 50, "high": 75, "extreme": 90}
    risk_value = risk_mapping.get(risk_level, 50)

    fig = go.Figure()

    # Main risk gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_value,
            domain={"x": [0, 1], "y": [0.5, 1]},
            title={"text": "Уровень риска"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {
                    "color": "darkred"
                    if risk_value > 70
                    else "orange"
                    if risk_value > 40
                    else "green"
                },
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
        )
    )

    # VaR indicators
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=var_1d,
            number={"suffix": "%"},
            title={"text": "VaR 1 день"},
            domain={"x": [0, 0.5], "y": [0, 0.4]},
        )
    )

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=var_5d,
            number={"suffix": "%"},
            title={"text": "VaR 5 дней"},
            domain={"x": [0.5, 1], "y": [0, 0.4]},
        )
    )

    fig.update_layout(
        title="⚠️ Оценка рисков портфеля",
        template="plotly_dark",
        height=500,
        font=dict(color="white"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_backtest_performance_chart(backtest: Dict[str, Any]) -> go.Figure:
    """Create a performance chart for backtest results."""

    # Extract or create sample performance data
    if "performance_data" in backtest:
        performance_data = backtest["performance_data"]
        df = pd.DataFrame(performance_data)
    else:
        # Create sample cumulative returns
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")  # 1 year
        daily_returns = np.random.normal(
            0.0008, 0.02, 252
        )  # ~20% annual return, 20% volatility
        cumulative_returns = (1 + pd.Series(daily_returns)).cumprod()

        # Create benchmark (market) returns
        benchmark_returns = np.random.normal(
            0.0005, 0.015, 252
        )  # ~12% annual return, 15% volatility
        benchmark_cumulative = (1 + pd.Series(benchmark_returns)).cumprod()

        df = pd.DataFrame(
            {
                "date": dates,
                "strategy_value": cumulative_returns * 100000,  # Starting with $100k
                "benchmark_value": benchmark_cumulative * 100000,
                "strategy_returns": daily_returns,
                "benchmark_returns": benchmark_returns,
            }
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Кумулятивная доходность", "Дневные доходности"),
        row_heights=[0.7, 0.3],
    )

    # Portfolio value line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["strategy_value"],
            mode="lines",
            name="Стратегия",
            line=dict(color="#00ff88", width=2),
        ),
        row=1,
        col=1,
    )

    # Benchmark line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["benchmark_value"],
            mode="lines",
            name="Бенчмарк",
            line=dict(color="#888888", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Daily returns
    colors = ["#00ff88" if r >= 0 else "#ff4444" for r in df["strategy_returns"]]
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["strategy_returns"] * 100,  # Convert to percentage
            name="Дневная доходность",
            marker_color=colors,
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    # Calculate key metrics for annotation
    total_return = (
        (df["strategy_value"].iloc[-1] / df["strategy_value"].iloc[0]) - 1
    ) * 100
    benchmark_return = (
        (df["benchmark_value"].iloc[-1] / df["benchmark_value"].iloc[0]) - 1
    ) * 100

    fig.update_layout(
        title=f"🔄 Результаты бэктеста - Общая доходность: {total_return:.1f}% vs Бенчмарк: {benchmark_return:.1f}%",
        template="plotly_dark",
        height=600,
        font=dict(color="white"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Format axes
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text="Стоимость портфеля ($)", row=1, col=1)
    fig.update_yaxes(title_text="Доходность (%)", row=2, col=1)

    return fig


def create_alpha_factors_heatmap(alpha_result: Dict[str, Any]) -> go.Figure:
    """Create a heatmap visualization for alpha factors IC."""

    reports = alpha_result.get("reports", [])

    if reports:
        # Extract factor names and IC values
        factors = []
        symbols = set()
        ic_matrix = {}

        for report in reports:
            factor_name = report.get("factor", "Unknown")
            factors.append(factor_name)

            # Get IC by symbol if available, otherwise use overall IC
            symbol_ics = report.get("symbol_ics", {})
            if symbol_ics:
                symbols.update(symbol_ics.keys())
                ic_matrix[factor_name] = symbol_ics
            else:
                # Use overall IC for all symbols
                overall_ic = report.get("ic1d", 0)
                ic_matrix[factor_name] = {"Overall": overall_ic}
                symbols.add("Overall")

        symbols = sorted(list(symbols))

        # Create matrix for heatmap
        z_matrix = []
        for factor in factors:
            row = []
            for symbol in symbols:
                ic_value = ic_matrix.get(factor, {}).get(symbol, 0)
                row.append(ic_value)
            z_matrix.append(row)
    else:
        # Sample data
        factors = ["Momentum", "Reversal", "Volume", "Volatility", "RSI"]
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        z_matrix = np.random.uniform(-0.2, 0.2, (len(factors), len(symbols))).tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=symbols,
            y=factors,
            colorscale=[
                [0, "#ff4444"],  # Red for negative IC
                [0.5, "#ffffff"],  # White for zero
                [1, "#00ff88"],  # Green for positive IC
            ],
            zmid=0,
            text=[[f"{val:.3f}" for val in row] for row in z_matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Information Coefficient"),
        )
    )

    fig.update_layout(
        title="🧮 Alpha Factory - Тепловая карта информационных коэффициентов",
        xaxis_title="Символы",
        yaxis_title="Альфа-факторы",
        template="plotly_dark",
        height=500,
        font=dict(color="white"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_sentiment_timeline(news: Dict[str, Any]) -> go.Figure:
    """Create a timeline chart showing sentiment over time."""

    news_data = news.get("news_data", [])

    if news_data and isinstance(news_data, list) and len(news_data) > 0:
        # Extract sentiment data from news
        all_articles = []
        for symbol_data in news_data:
            articles = symbol_data.get("articles", [])
            symbol = symbol_data.get("symbol", "Unknown")
            for article in articles:
                article["symbol"] = symbol
                all_articles.append(article)

        if all_articles:
            df = pd.DataFrame(all_articles)
            # Convert timestamps if available
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                # Create fake timestamps
                base_time = datetime.now() - timedelta(hours=24)
                df["timestamp"] = [
                    base_time + timedelta(hours=i) for i in range(len(df))
                ]
        else:
            df = pd.DataFrame()
    else:
        # Create sample sentiment timeline
        hours = 24
        timestamps = [datetime.now() - timedelta(hours=h) for h in range(hours, 0, -1)]
        sentiment_scores = np.random.normal(0.1, 0.3, hours)  # Slightly positive bias
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "sentiment_score": sentiment_scores,
                "symbol": ["MARKET"] * hours,
            }
        )

    if df.empty:
        # Fallback data
        timestamps = [datetime.now() - timedelta(hours=h) for h in range(24, 0, -1)]
        sentiment_scores = np.random.normal(0.1, 0.3, 24)
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "sentiment_score": sentiment_scores,
                "symbol": ["MARKET"] * 24,
            }
        )

    fig = go.Figure()

    # Color code by sentiment
    colors = [
        "#00ff88" if s > 0.1 else "#ff4444" if s < -0.1 else "#ffaa00"
        for s in df["sentiment_score"]
    ]

    # Scatter plot with sentiment scores
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["sentiment_score"],
            mode="markers+lines",
            name="Sentiment",
            marker=dict(color=colors, size=8, line=dict(width=1, color="white")),
            line=dict(color="rgba(255, 255, 255, 0.3)", width=1),
        )
    )

    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(
        y=0.2,
        line_dash="dot",
        line_color="green",
        opacity=0.3,
        annotation_text="Позитивный",
    )
    fig.add_hline(
        y=-0.2,
        line_dash="dot",
        line_color="red",
        opacity=0.3,
        annotation_text="Негативный",
    )

    fig.update_layout(
        title="📰 Анализ новостных настроений во времени",
        xaxis_title="Время",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1, 1]),
        template="plotly_dark",
        height=400,
        showlegend=False,
        font=dict(color="white"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_calibration_reliability_diagram(
    calibration_result: Dict[str, Any],
) -> go.Figure:
    """Create a reliability diagram for forecast calibration."""

    analysis = calibration_result.get("calibration_analysis", {})

    # Sample calibration data if not available
    if "calibration_bins" in analysis:
        bins = analysis["calibration_bins"]
        bin_boundaries = bins.get("bin_boundaries", [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        observed_frequencies = bins.get(
            "observed_frequencies", [0.1, 0.25, 0.45, 0.65, 0.85]
        )
        predicted_probabilities = bins.get(
            "predicted_probabilities", [0.1, 0.3, 0.5, 0.7, 0.9]
        )
        bin_counts = bins.get("bin_counts", [20, 25, 30, 15, 10])
    else:
        # Sample data showing calibration curve
        bin_boundaries = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        predicted_probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]
        # Realistic calibration: closer to diagonal = better calibrated
        observed_frequencies = [0.12, 0.28, 0.52, 0.68, 0.87]  # Slightly miscalibrated
        bin_counts = [20, 25, 30, 15, 10]

    fig = go.Figure()

    # Perfect calibration line (diagonal)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", dash="dash", width=2),
            hovertemplate="Perfect Calibration<extra></extra>",
        )
    )

    # Actual calibration curve
    fig.add_trace(
        go.Scatter(
            x=predicted_probabilities,
            y=observed_frequencies,
            mode="markers+lines",
            name="Model Calibration",
            marker=dict(
                size=[count / 2 for count in bin_counts],  # Size proportional to count
                color="#ff6b6b",
                line=dict(width=2, color="white"),
            ),
            line=dict(color="#ff6b6b", width=3),
            customdata=bin_counts,
            hovertemplate="Predicted: %{x:.2f}<br>Observed: %{y:.2f}<br>Count: %{customdata}<extra></extra>",
        )
    )

    # Calculate and display calibration error areas
    for i in range(len(predicted_probabilities)):
        x_val = predicted_probabilities[i]
        y_obs = observed_frequencies[i]
        y_perfect = x_val

        # Add error bars
        fig.add_shape(
            type="line",
            x0=x_val,
            y0=y_perfect,
            x1=x_val,
            y1=y_obs,
            line=dict(color="rgba(255, 107, 107, 0.3)", width=2),
        )

    # Calculate metrics for annotation
    ece = sum(
        abs(obs - pred) * count
        for obs, pred, count in zip(
            observed_frequencies, predicted_probabilities, bin_counts
        )
    ) / sum(bin_counts)

    fig.update_layout(
        title=f"📊 Диаграмма надежности прогнозов (ECE: {ece:.3f})",
        xaxis_title="Предсказанная вероятность",
        yaxis_title="Наблюдаемая частота",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_dark",
        height=500,
        font=dict(color="white"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
    )

    # Add annotation
    fig.add_annotation(
        x=0.05,
        y=0.95,
        text=f"ECE: {ece:.3f}<br>{'Хорошая калибровка' if ece < 0.1 else 'Требует улучшения'}",
        showarrow=False,
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="white",
        borderwidth=1,
    )

    return fig


def create_portfolio_allocation_pie(portfolio_data: Dict[str, Any]) -> go.Figure:
    """Create a pie chart for portfolio allocation."""

    if "positions" in portfolio_data and portfolio_data["positions"]:
        positions = portfolio_data["positions"]
        symbols = [pos.get("symbol", "Unknown") for pos in positions]
        values = [abs(pos.get("market_value", 0)) for pos in positions]
    else:
        # Sample portfolio data
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "Cash"]
        values = [25000, 20000, 18000, 15000, 12000, 10000]

    # Create pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=symbols,
                values=values,
                hole=0.4,  # Donut chart
                marker=dict(
                    colors=[
                        "#ff6b6b",
                        "#4ecdc4",
                        "#45b7d1",
                        "#96ceb4",
                        "#feca57",
                        "#fd79a8",
                    ]
                ),
                textinfo="label+percent",
                textposition="outside",
                hovertemplate="%{label}<br>Стоимость: $%{value:,.0f}<br>Доля: %{percent}<extra></extra>",
            )
        ]
    )

    # Add total value in center
    total_value = sum(values)
    fig.add_annotation(
        text=f"Общая<br>стоимость<br>${total_value:,.0f}",
        x=0.5,
        y=0.5,
        font_size=16,
        showarrow=False,
    )

    fig.update_layout(
        title="💼 Распределение портфеля",
        template="plotly_dark",
        height=400,
        font=dict(color="white"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# =============================================================================
# CHAINLIT HELPER FUNCTIONS
# =============================================================================


async def ensure_chat_session() -> str:
    """Ensure we have an active chat session and return session_id"""
    if CURRENT_SESSION["session_id"] is None:
        # Create new session with timestamp
        from datetime import datetime

        session_name = f"Trading Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        try:
            session_result = create_chat_session(
                session_name, "chainlit_user", "trading_analysis"
            )
            if session_result["success"]:
                CURRENT_SESSION["session_id"] = session_result["session_id"]
                CURRENT_SESSION["session_name"] = session_result["session_name"]

                # Send notification to user
                await cl.Message(
                    author="🧠 Память",
                    content=f"✅ Создана новая сессия: `{session_result['session_id']}`\n\n"
                    f"Все сообщения этого чата будут сохраняться в памяти агента для будущих сессий.",
                ).send()
            else:
                # Fallback session_id
                CURRENT_SESSION["session_id"] = "fallback_session"

        except Exception as e:
            # Fallback session_id
            CURRENT_SESSION["session_id"] = "fallback_session"
            await cl.Message(
                author="⚠️ Память",
                content=f"Не удалось создать сессию в памяти: {e}\n\nИспользуется временная сессия.",
            ).send()

    return CURRENT_SESSION["session_id"]


async def save_message_to_memory(
    content: str, message_type: str, metadata: Dict[str, Any] = None
) -> None:
    """Save message to SGR memory"""
    try:
        session_id = await ensure_chat_session()
        if session_id != "fallback_session":
            save_chat_message(session_id, message_type, content, metadata)
    except Exception as e:
        # Silent fail - memory is not critical for chat functionality
        pass


async def display_sgr_response(response: SGRTradingResponse, step_num: int) -> None:
    """Display SGR reasoning step beautifully."""

    # Main information as a table
    reasoning_table = f"""
**🎯 Текущее состояние:** {response.current_state}

**📋 Оставшиеся шаги:**
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(response.plan_remaining_steps_brief)])}

**🎬 Следующее действие:** `{response.function.tool}`

**📊 Статус:**
- Задача завершена: {"✅" if response.task_completed else "❌"}
    """

    msg = cl.Message(
        author=f"🧠 Шаг {step_num}",
        content=reasoning_table,
    )
    await msg.send()


async def display_tool_execution(tool_name: str, tool_params: Dict[str, Any]) -> None:
    """Display tool execution details."""

    params_str = json.dumps(tool_params, ensure_ascii=False, indent=2)

    tool_msg = f"""
**🔧 Выполнение инструмента:** `{tool_name}`

**📥 Параметры:**
```json
{params_str}
```
    """

    msg = cl.Message(
        author="🔧 Инструмент",
        content=tool_msg,
    )
    await msg.send()


async def display_tool_result(tool_name: str, result: Union[Dict, List, str]) -> None:
    """Display tool execution results beautifully."""

    if isinstance(result, dict):
        # Special handling for different result types
        if "market_data" in result:
            await display_market_data_result(result)
        elif "recommendation" in result:
            await display_trading_recommendation(result)
        elif "consensus_probability" in result:
            await display_forecast_result(result)
        elif "risk_level" in result:
            await display_risk_assessment(result)
        elif "sentiment_category" in result:
            await display_news_sentiment(result)
        elif "total_return" in result:
            await display_backtest_result(result)
        elif "url" in result and "sentiment_score" in result:
            await display_web_analysis_result(result)
        elif "search_query" in result and "sources_analyzed" in result:
            await display_web_research_result(result)
        elif "factors" in result and "reports" in result:
            await display_alpha_factors_result(result)
        elif "calibration_analysis" in result:
            await display_calibration_result(result)
        elif "risk_status" in result or "triggered_breakers" in result:
            await display_risk_controls_result(result)
        elif "calculated_metrics" in result and "requested_metrics" in result:
            await display_enhanced_metrics_result(result)
        elif "error" in result:
            await display_error_result(result)
        elif "final_answer_displayed" in result:
            # Skip - already displayed by ReportTaskCompletion
            return
        else:
            await display_json_result(tool_name, result)
    else:
        await display_text_result(tool_name, str(result))


async def display_market_data_result(data: Dict[str, Any]) -> None:
    """Display market data with interactive charts."""

    symbols_analyzed = data.get("symbols_analyzed", ["STOCK"])
    primary_symbol = symbols_analyzed[0] if symbols_analyzed else "STOCK"

    content = "**📊 Рыночные данные:**\n\n"
    content += f"**Символы:** {', '.join(symbols_analyzed)}\n"
    content += f"**Точек данных:** {data.get('data_points', 0)}\n"
    content += f"**Рыночный тренд:** {data.get('market_trend', 'unknown')}\n"
    content += f"**Волатильность:** {data.get('volatility_assessment', 'unknown')}\n\n"

    if data.get("key_insights"):
        content += "**🔍 Ключевые инсайты:**\n"
        for insight in data["key_insights"]:
            content += f"• {insight}\n"

    # Create interactive candlestick chart
    fig = create_candlestick_chart(data, primary_symbol)

    # Create chart element
    chart_element = cl.Plotly(
        name="market_data_chart", figure=fig, display="inline", size="large"
    )

    msg = cl.Message(
        author="📈 Рыночные данные", content=content, elements=[chart_element]
    )
    await msg.send()


async def display_trading_recommendation(rec: Dict[str, Any]) -> None:
    """Display trading recommendation beautifully."""

    symbol = rec.get("symbol", "")
    recommendation = rec.get("recommendation", "hold")
    confidence = rec.get("confidence_level", 0) * 100
    target_price = rec.get("target_price", 0)
    stop_loss = rec.get("stop_loss", 0)
    position_size = rec.get("position_size", 0)

    # Color code recommendation
    rec_color = (
        "🟢" if recommendation == "buy" else "🔴" if recommendation == "sell" else "🟡"
    )

    content = f"""
**{rec_color} Торговая рекомендация для {symbol}:**

**Действие:** {recommendation.upper()}
**Уверенность:** {confidence:.0f}%
**Целевая цена:** ${target_price:.2f}
**Стоп-лосс:** ${stop_loss:.2f}
**Размер позиции:** {position_size} акций

**📝 Обоснование:**
{rec.get('rationale', 'Нет обоснования')}

**⚠️ Уровень риска:** {rec.get('risk_level', 'unknown')}
**📊 Волатильность:** {rec.get('volatility', 0):.2%}
    """

    msg = cl.Message(
        author="🎯 Рекомендация",
        content=content,
    )
    await msg.send()


async def display_forecast_result(forecast: Dict[str, Any]) -> None:
    """Display forecast results with interactive probability chart."""

    question = forecast.get("question", "")
    probability = forecast.get("consensus_probability", 0.5) * 100
    confidence = forecast.get("confidence_score", 0) * 100
    disagreement = forecast.get("disagreement_level", 0)

    content = f"""
**🔮 Прогноз:**

**Вопрос:** {question}
**Вероятность:** {probability:.0f}%
**Уверенность:** {confidence:.0f}%
**Разногласия агентов:** {disagreement:.2f}

**Анализируемые символы:** {', '.join(forecast.get('symbols_analyzed', []))}
**Горизонт прогноза:** {forecast.get('forecast_horizon', 'unknown')}
    """

    # Show individual agent forecasts
    if forecast.get("individual_forecasts"):
        content += "\n**🤖 Прогнозы отдельных агентов:**\n"
        for f in forecast["individual_forecasts"]:
            agent = f.get("agent_type", "unknown")
            prob = f.get("prediction_probability", 0.5) * 100
            content += f"• **{agent}:** {prob:.0f}%\n"

    # Create interactive probability chart
    fig = create_forecast_probability_chart(forecast)

    chart_element = cl.Plotly(
        name="forecast_chart", figure=fig, display="inline", size="medium"
    )

    msg = cl.Message(author="🔮 Прогноз", content=content, elements=[chart_element])
    await msg.send()


async def display_risk_assessment(risk: Dict[str, Any]) -> None:
    """Display risk assessment with interactive gauge charts."""

    risk_level = risk.get("risk_level", "unknown")
    risk_emoji = (
        "🔴" if risk_level == "high" else "🟡" if risk_level == "medium" else "🟢"
    )

    content = f"""
**{risk_emoji} Оценка рисков:**

**Тип оценки:** {risk.get('assessment_type', 'unknown')}
**Уровень риска:** {risk_level}
**VaR (1 день):** {risk.get('var_1d', 0):.2%}
**VaR (5 дней):** {risk.get('var_5d', 0):.2%}
**Максимальная просадка:** {risk.get('max_drawdown', 0):.2%}
**Волатильность:** {risk.get('volatility', 0):.2%}
**Коэффициент Шарпа:** {risk.get('sharpe_ratio', 0):.2f}

**💡 Рекомендации:**
{chr(10).join([f"• {r}" for r in risk.get('recommendations', [])])}
    """

    # Create interactive risk gauge chart
    fig = create_risk_gauge_chart(risk)

    chart_element = cl.Plotly(
        name="risk_gauge", figure=fig, display="inline", size="large"
    )

    msg = cl.Message(
        author="⚠️ Оценка рисков", content=content, elements=[chart_element]
    )
    await msg.send()


async def display_news_sentiment(news: Dict[str, Any]) -> None:
    """Display news sentiment analysis with timeline chart."""

    sentiment = news.get("sentiment_category", "neutral")
    avg_sentiment = news.get("average_sentiment", 0)
    articles_count = news.get("news_articles", 0)
    data_source = news.get("data_source", "unknown")

    sentiment_emoji = (
        "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
    )

    content = f"""
**{sentiment_emoji} Анализ новостей:**

**Категория настроений:** {sentiment.upper()}
**Средний sentiment:** {avg_sentiment:.2f}
**Проанализировано статей:** {articles_count}
**Источник данных:** {data_source}
**Период анализа:** {news.get('lookback_hours', 24)} часов

**Анализируемые символы:** {', '.join(news.get('symbols_analyzed', []))}
    """

    # Show top news if available
    if news.get("news_data"):
        content += "\n**📰 Топ новости:**\n"
        for symbol_data in news["news_data"][:3]:
            symbol = symbol_data.get("symbol", "")
            articles = symbol_data.get("articles", [])
            if articles:
                content += f"\n**{symbol}:**\n"
                for article in articles[:2]:
                    title = article.get("title", "")
                    sentiment_score = article.get("sentiment_score", 0)
                    content += f"• {title} (sentiment: {sentiment_score:.2f})\n"

    # Create interactive sentiment timeline
    fig = create_sentiment_timeline(news)

    chart_element = cl.Plotly(
        name="sentiment_timeline", figure=fig, display="inline", size="medium"
    )

    msg = cl.Message(author="📰 Новости", content=content, elements=[chart_element])
    await msg.send()


async def display_backtest_result(backtest: Dict[str, Any]) -> None:
    """Display backtest results with performance charts."""

    total_return = backtest.get("total_return", 0) * 100
    win_rate = backtest.get("win_rate", 0) * 100
    sharpe_ratio = backtest.get("sharpe_ratio", 0)
    max_drawdown = backtest.get("max_drawdown", 0) * 100

    return_emoji = "🟢" if total_return > 0 else "🔴"

    content = f"""
**🔄 Результаты бэктеста:**

**Стратегия:** {backtest.get('strategy_name', 'unknown')}
**Период:** {backtest.get('period', 'unknown')}
**Символы:** {', '.join(backtest.get('symbols', []))}

**{return_emoji} Общая доходность:** {total_return:.1f}%
**Начальный капитал:** ${backtest.get('initial_capital', 0):,.0f}
**Конечный капитал:** ${backtest.get('final_capital', 0):,.0f}

**📊 Метрики производительности:**
- **Всего сделок:** {backtest.get('total_trades', 0)}
- **Win Rate:** {win_rate:.1f}%
- **Средняя доходность на сделку:** {backtest.get('avg_return_per_trade', 0):.2%}
- **Коэффициент Шарпа:** {sharpe_ratio:.2f}
- **Максимальная просадка:** {max_drawdown:.1f}%
- **Волатильность:** {backtest.get('volatility', 0):.1%}

**📝 Итог:** {backtest.get('performance_summary', 'Нет итога')}
    """

    # Create interactive performance chart
    fig = create_backtest_performance_chart(backtest)

    chart_element = cl.Plotly(
        name="backtest_performance", figure=fig, display="inline", size="large"
    )

    msg = cl.Message(author="🔄 Бэктест", content=content, elements=[chart_element])
    await msg.send()


async def display_web_analysis_result(analysis: Dict[str, Any]) -> None:
    """Display web content analysis results."""

    url = analysis.get("url", "")
    title = analysis.get("title", "")
    sentiment_score = analysis.get("sentiment_score", 0.0)
    content_type = analysis.get("content_type", "unknown")
    source_quality = analysis.get("source_quality", "unknown")
    credibility_score = analysis.get("credibility_score", 0.0)
    market_impact = analysis.get("market_impact_assessment", "neutral")

    # Sentiment emoji
    sentiment_emoji = (
        "🟢" if sentiment_score > 0.2 else "🔴" if sentiment_score < -0.2 else "🟡"
    )

    # Quality emoji
    quality_emoji = (
        "💎"
        if source_quality == "high"
        else "⭐"
        if source_quality == "medium"
        else "📄"
    )

    # Create progress bar for sentiment
    sentiment_normalized = (sentiment_score + 1) / 2  # Convert from [-1,1] to [0,1]
    progress_bar = "█" * int(sentiment_normalized * 15) + "░" * (
        15 - int(sentiment_normalized * 15)
    )

    content = f"""
**🌐 Анализ веб-контента**

**{quality_emoji} Источник:** [{title[:80]}...]({url})
**📋 Тип:** {content_type} | **🔍 Качество:** {source_quality} ({credibility_score:.1f})
**🌡️ Market Impact:** {market_impact}

**{sentiment_emoji} Sentiment:** {sentiment_score:+.3f}
`{progress_bar}`

**📊 Извлеченные данные:**
    """

    # Add financial metrics if available
    financial_metrics = analysis.get("key_financial_metrics", {})
    if financial_metrics:
        content += "\n**💰 Финансовые метрики:**\n"
        for metric, value in financial_metrics.items():
            if isinstance(value, (int, float)):
                if "price" in metric.lower() or "target" in metric.lower():
                    content += f"- **{metric}:** ${value:,.2f}\n"
                elif "revenue" in metric.lower() or "market_cap" in metric.lower():
                    if value >= 1_000_000_000:
                        content += f"- **{metric}:** ${value/1_000_000_000:.1f}B\n"
                    elif value >= 1_000_000:
                        content += f"- **{metric}:** ${value/1_000_000:.1f}M\n"
                    else:
                        content += f"- **{metric}:** ${value:,.0f}\n"
                else:
                    content += f"- **{metric}:** {value}\n"

    # Add extracted insights
    extracted_data = analysis.get("extracted_data", {})
    if extracted_data:
        if "price_targets" in extracted_data and extracted_data["price_targets"]:
            content += "\n**🎯 Прогнозы аналитиков:**\n"
            for target in extracted_data["price_targets"][:3]:
                analyst = target.get("analyst", "Unknown")
                price = target.get("price_target", 0)
                action = target.get("action", "set")
                content += f"- **{analyst}:** {action} target to ${price:.2f}\n"

    msg = cl.Message(
        author="🌐 Веб-анализ",
        content=content,
    )
    await msg.send()


async def display_web_research_result(research: Dict[str, Any]) -> None:
    """Display web research results with detailed source visualization."""

    search_query = research.get("search_query", "")
    sources_analyzed = research.get("sources_analyzed", 0)
    overall_sentiment = research.get("overall_sentiment", 0.0)
    research_depth = research.get("research_depth", "unknown")
    sentiment_range = research.get("sentiment_range", [0, 0])

    sentiment_emoji = (
        "🟢" if overall_sentiment > 0.2 else "🔴" if overall_sentiment < -0.2 else "🟡"
    )

    # Create progress bar for sentiment
    sentiment_normalized = (overall_sentiment + 1) / 2  # Convert from [-1,1] to [0,1]
    progress_bar = "█" * int(sentiment_normalized * 20) + "░" * (
        20 - int(sentiment_normalized * 20)
    )

    content = f"""
**🔍 Комплексное веб-исследование**

**🎯 Запрос:** `{search_query}`
**📊 Глубина:** {research_depth} | **📈 Источников:** {sources_analyzed}

**{sentiment_emoji} Общий sentiment:** {overall_sentiment:.3f}
`{progress_bar}` ({sentiment_range[0]:.2f} ↔ {sentiment_range[1]:.2f})
    """

    # Add key findings
    key_findings = research.get("key_findings", [])
    if key_findings:
        content += "\n**🔑 Ключевые находки:**\n"
        for finding in key_findings[:5]:
            content += f"• {finding}\n"

    # Add financial consensus
    financial_consensus = research.get("financial_consensus", {})
    if financial_consensus:
        content += "\n**💰 Финансовый консенсус:**\n"
        for metric, data in financial_consensus.items():
            if isinstance(data, dict) and "average" in data:
                avg = data["average"]
                consensus = data.get("consensus", "neutral")
                consensus_emoji = (
                    "🟢"
                    if consensus == "bullish"
                    else "🔴"
                    if consensus == "bearish"
                    else "🟡"
                )

                if "price" in metric.lower():
                    content += f"- **{metric}:** ${avg:.2f} {consensus_emoji}\n"
                elif "revenue" in metric.lower() or "cap" in metric.lower():
                    if avg >= 1_000_000_000:
                        content += f"- **{metric}:** ${avg/1_000_000_000:.1f}B {consensus_emoji}\n"
                    elif avg >= 1_000_000:
                        content += (
                            f"- **{metric}:** ${avg/1_000_000:.1f}M {consensus_emoji}\n"
                        )
                    else:
                        content += f"- **{metric}:** ${avg:,.0f} {consensus_emoji}\n"

    # Add risk factors
    risk_factors = research.get("risk_factors", [])
    if risk_factors:
        content += "\n**⚠️ Факторы риска:**\n"
        for risk in risk_factors[:3]:
            content += f"• {risk}\n"

    # Add opportunities
    opportunities = research.get("opportunities", [])
    if opportunities:
        content += "\n**💡 Возможности:**\n"
        for opportunity in opportunities[:3]:
            content += f"• {opportunity}\n"

    # Add analyzed sources with links
    analyzed_sources = research.get("analyzed_sources", [])
    if analyzed_sources:
        content += "\n**🌐 Проанализированные источники:**\n"
        for i, source in enumerate(analyzed_sources[:5], 1):
            url = source.get("url", "")
            title = source.get("title", "Без названия")
            sentiment = source.get("sentiment_score", 0.0)
            source_quality = source.get("source_quality", "medium")

            # Sentiment emoji for each source
            source_emoji = (
                "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "🟡"
            )

            # Quality badge
            quality_badge = {"high": "⭐", "medium": "🔶", "low": "⚠️"}.get(
                source_quality, "🔶"
            )

            # Truncate title for display
            display_title = title[:60] + "..." if len(title) > 60 else title

            if url:
                content += f"{i}. {quality_badge} [{display_title}]({url}) {source_emoji} ({sentiment:+.2f})\n"
            else:
                content += f"{i}. {quality_badge} {display_title} {source_emoji} ({sentiment:+.2f})\n"

    # Add source quality distribution
    quality_dist = research.get("source_quality_distribution", {})
    if quality_dist:
        content += "\n**📊 Качество источников:**\n"
        total = sum(quality_dist.values())
        if total > 0:
            high_pct = (quality_dist.get("high", 0) / total) * 100
            medium_pct = (quality_dist.get("medium", 0) / total) * 100
            low_pct = (quality_dist.get("low", 0) / total) * 100

            content += f"⭐ Высокое: {quality_dist.get('high', 0)} ({high_pct:.0f}%)\n"
            content += (
                f"🔶 Среднее: {quality_dist.get('medium', 0)} ({medium_pct:.0f}%)\n"
            )
            content += f"⚠️ Низкое: {quality_dist.get('low', 0)} ({low_pct:.0f}%)\n"

    msg = cl.Message(
        author="🔍 Исследование",
        content=content,
    )
    await msg.send()


async def display_alpha_factors_result(alpha_result: Dict[str, Any]) -> None:
    """Display alpha factors generation results with heatmap visualization."""

    success = alpha_result.get("success", False)
    factors = alpha_result.get("factors", [])
    reports = alpha_result.get("reports", [])
    symbols_processed = alpha_result.get("symbols_processed", 0)
    factors_computed = alpha_result.get("factors_computed", 0)

    if not success:
        error_msg = alpha_result.get("error", "Unknown error")
        content = f"""
**❌ Ошибка генерации альфа-факторов:**

{error_msg}
        """

        await cl.Message(
            author="🧮 Alpha Factory",
            content=content,
        ).send()
        return

    # Main results summary
    content = f"""
**🧮 Alpha Factory - Генерация факторов завершена**

**📊 Статистика:**
- **Символов обработано:** {symbols_processed}
- **Факторов вычислено:** {factors_computed}
- **Временных рядов создано:** {len(factors)}
- **Отчетов сгенерировано:** {len(reports)}
    """

    # Information Coefficient (IC) analysis
    if reports:
        content += "\n**📈 Анализ информационного коэффициента (IC):**\n"

        # Sort reports by absolute IC value
        valid_reports = []
        weak_reports = []

        for report in reports:
            factor_name = report.get("factor", "Unknown")
            ic = report.get("ic1d")
            coverage = report.get("coverage", 0)

            if ic is not None:
                abs_ic = abs(ic)
                if abs_ic > 0.05:  # Meaningful IC
                    valid_reports.append((factor_name, ic, coverage, abs_ic))
                else:
                    weak_reports.append((factor_name, ic, coverage))
            else:
                weak_reports.append((factor_name, None, coverage))

        # Sort by absolute IC (strongest first)
        valid_reports.sort(key=lambda x: x[3], reverse=True)

        # Display strong factors
        if valid_reports:
            content += "\n**🏆 Сильные предикторы (|IC| > 0.05):**\n"
            for factor_name, ic, coverage, abs_ic in valid_reports:
                strength = (
                    "Очень сильный"
                    if abs_ic > 0.15
                    else "Сильный"
                    if abs_ic > 0.10
                    else "Умеренный"
                )
                direction = "📈 Положительный" if ic > 0 else "📉 Отрицательный"

                # Create progress bar for IC strength
                ic_normalized = min(abs_ic * 5, 1.0)  # Scale to 0-1 for display
                progress_bar = "█" * int(ic_normalized * 10) + "░" * (
                    10 - int(ic_normalized * 10)
                )

                content += f"• **{factor_name}**: IC={ic:.4f} ({coverage} obs)\n"
                content += f"  {direction} | {strength} | `{progress_bar}`\n"

        # Display weak factors
        if weak_reports:
            content += "\n**📊 Слабые факторы (|IC| ≤ 0.05 или N/A):**\n"
            for factor_name, ic, coverage in weak_reports[
                :3
            ]:  # Show max 3 weak factors
                if ic is not None:
                    content += f"• **{factor_name}**: IC={ic:.4f} ({coverage} obs) - Слабый сигнал\n"
                else:
                    content += f"• **{factor_name}**: IC=N/A ({coverage} obs) - Недостаточно данных\n"

    # Alpha Factory info
    content += f"""

**ℹ️ Alpha Factory Info:**
- Использованы операторы WorldQuant Finding Alphas
- Кросс-секционный анализ с ранговой корреляцией
- Временной горизонт: {alpha_result.get('timestamp', 'N/A')[:19] if alpha_result.get('timestamp') else 'N/A'}
    """

    # Create interactive heatmap for alpha factors
    fig = create_alpha_factors_heatmap(alpha_result)

    chart_element = cl.Plotly(
        name="alpha_factors_heatmap", figure=fig, display="inline", size="large"
    )

    await cl.Message(
        author="🧮 Alpha Factory", content=content, elements=[chart_element]
    ).send()


async def display_calibration_result(calibration_result: Dict[str, Any]) -> None:
    """Display forecast calibration analysis with reliability diagram."""

    analysis = calibration_result.get("calibration_analysis", {})
    success = calibration_result.get("success", False)

    if not success or "error" in analysis:
        error_msg = analysis.get("error", "Unknown error")
        content = f"""
**❌ Ошибка анализа калибровки:**

{error_msg}
        """
        await cl.Message(
            author="📊 Калибровка прогнозов",
            content=content,
        ).send()
        return

    # Main calibration summary
    forecast_counts = analysis.get("forecast_counts", {})
    metrics = analysis.get("calibration_metrics", {})
    agent_performance = analysis.get("agent_performance", {})

    content = f"""
**📊 Анализ калибровки прогнозов**

**📈 Статистика прогнозов:**
- **Всего зарегистрировано:** {forecast_counts.get('total_registered', 0)}
- **Разрешено:** {forecast_counts.get('total_resolved', 0)}
- **За период:** {forecast_counts.get('recent_resolved', 0)}

**🎯 Метрики калибровки:**
- **Brier Score:** {metrics.get('brier_score', 0):.4f} (чем меньше, тем лучше)
- **Skill Score:** {metrics.get('brier_skill_score', 0):.4f if metrics.get('brier_skill_score') else 'N/A'} (vs baseline)
- **Expected Calibration Error:** {metrics.get('expected_calibration_error', 0):.4f}
- **Max Calibration Error:** {metrics.get('max_calibration_error', 0):.4f}

**📊 Качество калибровки:** {analysis.get('calibration_quality', 'Unknown')}
    """

    # Agent performance breakdown
    if agent_performance:
        content += "\n**🤖 Производительность агентов:**\n"
        for agent, perf in agent_performance.items():
            brier = perf.get("brier_score", 0)
            count = perf.get("count", 0)

            # Quality indicator
            quality_emoji = "🟢" if brier < 0.2 else "🟡" if brier < 0.3 else "🔴"

            content += f"- **{agent}:** {quality_emoji} Brier {brier:.4f} ({count} прогнозов)\n"

    # Recommendations
    recommendations = calibration_result.get("recommendations", [])
    if recommendations:
        content += "\n**💡 Рекомендации:**\n"
        for rec in recommendations:
            content += f"• {rec}\n"

    # Period info
    content += f"\n**ℹ️ Период анализа:** {calibration_result.get('time_period', 'N/A')}"

    # Create reliability diagram
    fig = create_calibration_reliability_diagram(calibration_result)

    chart_element = cl.Plotly(
        name="calibration_diagram", figure=fig, display="inline", size="large"
    )

    await cl.Message(
        author="📊 Калибровка прогнозов", content=content, elements=[chart_element]
    ).send()


async def display_risk_controls_result(risk_result: Dict[str, Any]) -> None:
    """Display risk controls management results."""

    action = risk_result.get("action", "unknown")
    success = risk_result.get("success", False)

    if not success:
        error_msg = risk_result.get("error", "Unknown error")
        content = f"""
**❌ Ошибка управления рисками:**

{error_msg}
        """
        await cl.Message(
            author="⚠️ Управление рисками",
            content=content,
        ).send()
        return

    if action == "status":
        # Risk status display
        risk_status = risk_result.get("risk_status", {})

        content = f"""
**⚠️ Статус системы управления рисками**

**🛑 Kill Switch:** {'🔴 АКТИВЕН' if risk_status.get('kill_switch_active') else '🟢 Неактивен'}
        """

        if risk_status.get("kill_switch_reason"):
            content += f"**Причина:** {risk_status['kill_switch_reason']}\n"

        # Circuit breakers
        breakers = risk_status.get("circuit_breakers", {})
        if breakers:
            content += "\n**⚡ Circuit Breakers:**\n"
            for breaker_id, breaker in breakers.items():
                status_emoji = "🔴" if breaker.get("triggered") else "🟢"
                utilization = breaker.get("utilization_pct", 0)

                content += (
                    f"- **{breaker.get('name')}:** {status_emoji} {utilization:.1f}%\n"
                )
                content += f"  Текущее: {breaker.get('current_value', 0):.4f} | Лимит: {breaker.get('threshold', 0):.4f}\n"

        # Risk limits
        limits = risk_status.get("risk_limits", {})
        if limits:
            content += "\n**📊 Лимиты риска:**\n"
            for limit_id, limit in limits.items():
                utilization = limit.get("utilization_pct", 0)
                breach_count = limit.get("breach_count", 0)

                status_emoji = (
                    "🔴" if utilization > 80 else "🟡" if utilization > 60 else "🟢"
                )

                content += (
                    f"- **{limit.get('type')}:** {status_emoji} {utilization:.1f}%\n"
                )
                content += f"  Нарушений: {breach_count}\n"

        content += f"\n**📈 Решений за 24ч:** {risk_status.get('recent_decisions', 0)}"

    elif action == "trigger_test":
        # Circuit breaker test results
        triggered = risk_result.get("triggered_breakers", [])
        test_metrics = risk_result.get("test_metrics", {})

        content = f"""
**🔥 Тест Circuit Breakers**

**Результат:** {len(triggered)} сработали из {len(triggered) + 0} проверенных

**📊 Тестовые метрики:**
- **Просадка:** {test_metrics.get('current_drawdown', 0):.1%}
- **VaR:** {test_metrics.get('current_var', 0):.1%}  
- **Дневной P&L:** ${test_metrics.get('daily_pnl', 0):,.0f}
- **Волатильность:** {test_metrics.get('current_volatility', 0):.1%}

**⚡ Сработавшие автоматы:** {len(triggered)}
        """

        for breaker_id in triggered:
            content += f"- {breaker_id}\n"

    else:
        # Other actions
        message = risk_result.get("message", "Action completed")
        content = f"""
**⚠️ Управление рисками - {action}**

**Результат:** {message}
        """

    await cl.Message(
        author="⚠️ Управление рисками",
        content=content,
    ).send()


async def display_enhanced_metrics_result(metrics_result: Dict[str, Any]) -> None:
    """Display enhanced trading metrics results."""

    success = metrics_result.get("success", False)

    if not success:
        error_msg = metrics_result.get("error", "Unknown error")
        content = f"""
**❌ Ошибка вычисления метрик:**

{error_msg}
        """
        await cl.Message(
            author="📈 Улучшенные метрики",
            content=content,
        ).send()
        return

    requested_metrics = metrics_result.get("requested_metrics", [])
    calculated_metrics = metrics_result.get("calculated_metrics", {})
    time_period = metrics_result.get("time_period", "1y")
    benchmark = metrics_result.get("benchmark")

    content = f"""
**📈 Улучшенные торговые метрики**

**⏱️ Период:** {time_period}
**📊 Бенчмарк:** {benchmark or 'Нет'}
**🔢 Запрошенные метрики:** {', '.join(requested_metrics)}
    """

    # Calmar Ratio
    if "calmar_ratio" in calculated_metrics:
        calmar = calculated_metrics["calmar_ratio"]
        ratio = calmar.get("ratio", 0)
        annual_return = calmar.get("annual_return", 0)
        max_drawdown = calmar.get("max_drawdown", 0)

        # Quality assessment
        quality_emoji = (
            "🏆"
            if ratio > 2.0
            else "✅"
            if ratio > 1.0
            else "⚠️"
            if ratio > 0.5
            else "❌"
        )

        content += f"""

**📊 Calmar Ratio:** {quality_emoji} {ratio:.3f}
- **Годовая доходность:** {annual_return:.2%}
- **Максимальная просадка:** {max_drawdown:.2%}
        """

    # Sortino Ratio
    if "sortino_ratio" in calculated_metrics:
        sortino = calculated_metrics["sortino_ratio"]
        ratio = sortino.get("ratio", 0)
        annual_return = sortino.get("annual_return", 0)
        downside_deviation = sortino.get("downside_deviation", 0)

        quality_emoji = "🏆" if ratio > 2.0 else "✅" if ratio > 1.0 else "📉"

        content += f"""

**📊 Sortino Ratio:** {quality_emoji} {ratio:.3f}
- **Годовая доходность:** {annual_return:.2%}
- **Downside Deviation:** {downside_deviation:.2%}
        """

    # Implementation Shortfall
    if "implementation_shortfall" in calculated_metrics:
        impl = calculated_metrics["implementation_shortfall"]
        shortfall_bps = impl.get("shortfall_bps", 0)
        market_impact = impl.get("market_impact_bps", 0)
        timing_cost = impl.get("timing_cost_bps", 0)
        execution_time = impl.get("execution_time_seconds", 0)

        quality_emoji = (
            "🎯"
            if abs(shortfall_bps) < 10
            else "✅"
            if abs(shortfall_bps) < 25
            else "⚠️"
        )

        content += f"""

**🎯 Implementation Shortfall:** {quality_emoji} {shortfall_bps:.1f} bps
- **Market Impact:** {market_impact:.1f} bps
- **Timing Cost:** {timing_cost:.1f} bps  
- **Среднее время исполнения:** {execution_time:.1f}s
        """

    # Recommendations
    recommendations = metrics_result.get("recommendations", [])
    if recommendations:
        content += "\n\n**💡 Рекомендации:**\n"
        for rec in recommendations:
            content += f"• {rec}\n"

    # Comprehensive analysis if available
    if "comprehensive_analysis" in metrics_result:
        comp = metrics_result["comprehensive_analysis"]["additional_metrics"]

        content += f"""

**📊 Дополнительные метрики:**
- **Sharpe Ratio:** {comp.get('sharpe_ratio', 0):.3f}
- **Win Rate:** {comp.get('win_rate', 0):.1%}
- **Profit Factor:** {comp.get('profit_factor', 0):.2f}
- **Волатильность:** {comp.get('volatility', 0):.1%}
        """

    await cl.Message(
        author="📈 Улучшенные метрики",
        content=content,
    ).send()


async def display_error_result(error: Dict[str, Any]) -> None:
    """Display error result."""

    content = f"""
**❌ Ошибка:**

**Инструмент:** {error.get('tool', 'Unknown')}
**Описание:** {error.get('error', 'Unknown error')}
    """

    msg = cl.Message(
        author="❌ Ошибка",
        content=content,
    )
    await msg.send()


async def display_json_result(tool_name: str, result: Dict[str, Any]) -> None:
    """Display JSON result."""

    result_str = json.dumps(result, ensure_ascii=False, indent=2)

    content = f"""
**🔍 Результат {format_tool_name(tool_name)}:**

```json
{result_str}
```
    """

    msg = cl.Message(
        author="📄 Результат",
        content=content,
    )
    await msg.send()


async def display_text_result(tool_name: str, result: str) -> None:
    """Display text result."""

    content = f"""
**📝 Результат {format_tool_name(tool_name)}:**

{result}
    """

    msg = cl.Message(
        author="📝 Результат",
        content=content,
    )
    await msg.send()


async def display_final_answer(completion: ReportTaskCompletion) -> None:
    """Display final answer and recommendations with portfolio visualization."""

    content = "**🎯 ИТОГОВЫЙ АНАЛИЗ:**\n\n"

    if completion.final_recommendation:
        content += (
            f"**💡 Торговая рекомендация:**\n{completion.final_recommendation}\n\n"
        )

    content += "**✅ Выполненные шаги:**\n"
    for i, step in enumerate(completion.completed_steps_laconic, 1):
        content += f"{i}. {step}\n"

    elements = []

    # Show portfolio status if available
    if DB.memory.positions:
        content += "\n**💼 Текущие позиции:**\n"
        for position in DB.memory.positions[-5:]:
            content += f"• {position.symbol}: {position.quantity} акций @ ${position.avg_cost:.2f}\n"

        # Create portfolio visualization
        portfolio_data = {
            "positions": [
                {"symbol": pos.symbol, "market_value": pos.quantity * pos.avg_cost}
                for pos in DB.memory.positions
            ]
        }

        fig = create_portfolio_allocation_pie(portfolio_data)
        chart_element = cl.Plotly(
            name="portfolio_allocation", figure=fig, display="inline", size="medium"
        )
        elements.append(chart_element)

    msg = cl.Message(
        author="🏁 Завершение", content=content, elements=elements if elements else None
    )
    await msg.send()


# =============================================================================
# TOOL EXECUTION FUNCTIONS
# =============================================================================


@cl.step(type="tool")
async def execute_tool(
    tool_name: str, tool_params: Dict[str, Any]
) -> Union[Dict, List, str]:
    """Execute financial trading tool and return result."""

    try:
        # Display tool execution
        await display_tool_execution(tool_name, tool_params)

        # Execute based on tool name
        if tool_name == "get_market_data":
            cmd = MarketDataRequest(**tool_params)
            result = get_market_data(cmd.symbols, cmd.timeframe, cmd.period)

        elif tool_name == "analyze_trading_opportunity":
            cmd = TradingAnalysisRequest(**tool_params)
            result = analyze_trading_opportunity(
                cmd.symbol, cmd.analysis_type, cmd.budget, cmd.risk_tolerance
            )

        elif tool_name == "generate_forecast":
            cmd = ForecastRequest(**tool_params)
            result = generate_forecast(
                cmd.question, cmd.symbols, cmd.forecast_horizon, cmd.agent_types
            )

        elif tool_name == "assess_risk":
            cmd = RiskAssessmentRequest(**tool_params)
            result = assess_risk(cmd.symbol, cmd.trade_amount, cmd.assessment_type)

        elif tool_name == "analyze_news":
            cmd = NewsAnalysisRequest(**tool_params)
            result = analyze_news(cmd.symbols, cmd.sources, cmd.lookback_hours)

        elif tool_name == "run_backtest":
            cmd = BacktestRequest(**tool_params)
            result = run_backtest(
                cmd.strategy_name,
                cmd.symbols,
                cmd.start_date,
                cmd.end_date,
                cmd.initial_capital,
            )

        elif tool_name == "analyze_web_content":
            cmd = WebAnalysisRequest(**tool_params)
            result = analyze_web_content(
                cmd.url,
                cmd.analysis_type,
                cmd.extract_data_points,
                cmd.include_links,
            )

        elif tool_name == "research_financial_topic":
            cmd = ComprehensiveWebResearch(**tool_params)

            # Show research progress
            progress_msg = cl.Message(
                author="🔍 Исследование",
                content=f"🔎 **Начинаю комплексное исследование...**\n\n"
                f"**Запрос:** `{cmd.search_query}`\n"
                f"**Максимум источников:** {cmd.max_sources}\n"
                f"**Глубина:** {cmd.research_depth}\n\n"
                f"⏳ Поиск и анализ источников в процессе...",
            )
            await progress_msg.send()

            result = research_financial_topic(
                cmd.search_query,
                cmd.research_depth,
                cmd.max_sources,
                cmd.include_news,
                cmd.include_analyst_reports,
                cmd.extract_financial_data,
                cmd.time_range,
            )

        elif tool_name == "generate_alphas":
            cmd = AlphaGenerationRequest(**tool_params)

            # Show alpha generation progress
            progress_msg = cl.Message(
                author="🧮 Alpha Factory",
                content=f"🔬 **Начинаю генерацию альфа-факторов...**\n\n"
                f"**Символы:** {', '.join(cmd.symbols)}\n"
                f"**Количество факторов:** {len(cmd.specs)}\n"
                f"**Период данных:** {cmd.period}\n"
                f"**Таймфрейм:** {cmd.timeframe}\n\n"
                f"⏳ Скачивание данных и вычисление факторов...",
            )
            await progress_msg.send()

            result = generate_alphas(cmd.symbols, cmd.specs, cmd.timeframe, cmd.period)

        elif tool_name == "analyze_calibration":
            cmd = CalibrationAnalysisRequest(**tool_params)

            progress_msg = cl.Message(
                author="📊 Калибровка прогнозов",
                content=f"🔍 **Анализ калибровки прогнозов...**\n\n"
                f"**Период:** {cmd.time_period}\n"
                f"**Агенты:** {', '.join(cmd.agent_types)}\n"
                f"**Диаграммы:** {'Да' if cmd.include_reliability_diagram else 'Нет'}\n\n"
                f"⏳ Вычисление Brier Score и ECE...",
            )
            await progress_msg.send()

            result = analyze_calibration(
                cmd.time_period, cmd.agent_types, cmd.include_reliability_diagram
            )

        elif tool_name == "manage_risk_controls":
            cmd = RiskControlsRequest(**tool_params)

            progress_msg = cl.Message(
                author="⚠️ Управление рисками",
                content=f"⚡ **Управление системой рисков...**\n\n"
                f"**Действие:** {cmd.action}\n"
                f"**Тип контроля:** {cmd.control_type or 'Все'}\n\n"
                f"⏳ Выполнение операции...",
            )
            await progress_msg.send()

            result = manage_risk_controls_wrapper(
                cmd.action,
                cmd.control_type,
                reason=cmd.reason,
                authorized_by=cmd.authorized_by,
                breaker_id=cmd.breaker_id,
                activate=cmd.activate,
                threshold_value=cmd.threshold_value,
                trigger_type=cmd.trigger_type,
                name=cmd.name,
            )

        elif tool_name == "calculate_enhanced_metrics":
            cmd = EnhancedMetricsRequest(**tool_params)

            progress_msg = cl.Message(
                author="📈 Улучшенные метрики",
                content=f"📊 **Вычисление торговых метрик...**\n\n"
                f"**Метрики:** {', '.join(cmd.metrics)}\n"
                f"**Период:** {cmd.time_period}\n"
                f"**Бенчмарк:** {cmd.benchmark or 'Нет'}\n\n"
                f"⏳ Расчет Calmar, Sortino, Implementation Shortfall...",
            )
            await progress_msg.send()

            result = calculate_enhanced_metrics_wrapper(
                cmd.metrics, cmd.time_period, cmd.benchmark
            )

        elif tool_name == "report_completion":
            cmd = ReportTaskCompletion(**tool_params)
            await display_final_answer(cmd)
            return {
                "completed_steps": cmd.completed_steps_laconic,
                "final_recommendation": cmd.final_recommendation or "",
                "status": cmd.code,
                "final_answer_displayed": True,
            }

        else:
            return {"error": f"Unknown tool: {tool_name}"}

        return result

    except Exception as e:
        return {"error": str(e), "tool": tool_name}


# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================


async def run_sgr_step(task: str, conversation_log: List[Dict]) -> SGRTradingResponse:
    """Execute one SGR reasoning step."""

    deployment_name = settings.azure_openai_deployment_name

    # Prepare system prompt
    system_prompt = f"""Вы - финансовый ассистент с Schema-Guided Reasoning.

Инструменты:
- get_market_data: Рыночные данные (symbols, timeframe, period)
- analyze_trading_opportunity: Анализ торговых возможностей
- generate_forecast: Вероятностные прогнозы
- assess_risk: Оценка рисков с VaR
- analyze_news: Анализ новостей
- run_backtest: Валидация стратегии
- analyze_web_content: Веб-скрапинг финансовых данных
- research_financial_topic: Исследование (макс. 5 источников)
- generate_alphas: Генерация альфа-факторов WorldQuant (delta, zscore, ts_rank, decay_linear, ts_mean, ts_std, delay) с расчетом информационного коэффициента
- analyze_calibration: Анализ калибровки прогнозов (Brier Score, ECE, Reliability Diagrams)
- manage_risk_controls: Управление рисками (circuit breakers, kill-switch, лимиты)
- calculate_enhanced_metrics: Расчет продвинутых метрик (Calmar, Sortino, Implementation Shortfall)
- report_completion: Финальные рекомендации
- get_chat_history: Получить историю чата (session_id, limit, message_types)
- save_chat_message: Сохранить сообщение (session_id, message_type, content)
- create_trading_rule: Создать торговое правило (description, type, parameters)
- get_trading_memory: Получить память о правилах (memory_type, filter_by)

Задача: {task}

Риски: Макс позиция {settings.max_position_size:.1%}, макс просадка {settings.max_daily_drawdown:.1%}
Режим: {"Бумажная торговля" if settings.enable_paper_trading else "Реальная торговля"}

Алгоритм:
1. Собрать рыночные данные
2. Провести анализ
3. Оценить риски
4. Дать рекомендации через report_completion

ОТВЕЧАЙТЕ НА РУССКОМ."""

    messages = [{"role": "system", "content": system_prompt}] + conversation_log

    # Calculate token limits dynamically based on model capacity
    max_completion_tokens = settings.max_completion_tokens

    # If adaptive tokens enabled, adjust based on current prompt size
    if settings.enable_adaptive_tokens:
        # Estimate prompt tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_prompt_tokens = (
            sum(len(msg.get("content", "")) for msg in messages) // 4
        )

        # Ensure we don't exceed the total model capacity (200K)
        available_tokens = 200000 - estimated_prompt_tokens - 1000  # Buffer for safety
        max_completion_tokens = min(max_completion_tokens, max(10000, available_tokens))

    # Use structured output
    completion = await client.beta.chat.completions.parse(
        model=deployment_name,
        response_format=SGRTradingResponse,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )

    try:
        # Check if completion has choices and they are valid
        if not completion.choices or len(completion.choices) == 0:
            print("Warning: No choices in completion, creating fallback response")
            from models import ReportTaskCompletion

            return SGRTradingResponse(
                current_state="No completion choices available, using fallback",
                plan_remaining_steps_brief=["Continue analysis with available tools"],
                task_completed=False,
                function=ReportTaskCompletion(
                    completed_steps_laconic=["No completion choices, continuing"],
                    code="no_choices_error",
                ),
            )

        # Check if message exists and has parsed content
        choice = completion.choices[0]
        if not hasattr(choice, "message") or choice.message is None:
            print("Warning: No message in choice, creating fallback response")
            from models import ReportTaskCompletion

            return SGRTradingResponse(
                current_state="No message in completion choice, using fallback",
                plan_remaining_steps_brief=["Continue analysis with available tools"],
                task_completed=False,
                function=ReportTaskCompletion(
                    completed_steps_laconic=["No message content, continuing"],
                    code="no_message_error",
                ),
            )

        sgr_response = choice.message.parsed

        # Log token usage for debugging
        if hasattr(completion, "usage") and completion.usage:
            usage = completion.usage
            print(
                f"Token usage: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}"
            )

        # Check if sgr_response is None
        if sgr_response is None:
            print("Warning: SGR response is None, creating fallback response")
            from models import ReportTaskCompletion

            return SGRTradingResponse(
                current_state="Parsing error occurred, continuing with fallback response",
                plan_remaining_steps_brief=[
                    "Continue analysis with next available tool"
                ],
                task_completed=False,
                function=ReportTaskCompletion(
                    completed_steps_laconic=[
                        "Encountered parsing issue, continuing analysis"
                    ],
                    code="parsing_error",
                ),
            )

        return sgr_response

    except Exception as e:
        error_msg = str(e)

        # Handle token limit errors more gracefully
        if any(
            phrase in error_msg
            for phrase in [
                "length limit was reached",
                "CompletionUsage",
                "max_tokens",
                "token limit",
            ]
        ):
            print(f"Token limit reached, retrying with reduced tokens: {e}")

            try:
                # Retry with 50% fewer tokens
                reduced_tokens = max(5000, max_completion_tokens // 2)
                print(f"Retrying with reduced tokens: {reduced_tokens}")

                retry_completion = await client.beta.chat.completions.parse(
                    model=deployment_name,
                    response_format=SGRTradingResponse,
                    messages=messages,
                    max_completion_tokens=reduced_tokens,
                )

                return retry_completion.choices[0].message.parsed

            except Exception as retry_error:
                print(f"Retry with reduced tokens also failed: {retry_error}")
                # Return a simplified response
                from models import ReportTaskCompletion

                return SGRTradingResponse(
                    current_state="Token limit exceeded, continuing with reduced response",
                    plan_remaining_steps_brief=["Continue analysis with next tool"],
                    task_completed=False,
                    function=ReportTaskCompletion(
                        completed_steps_laconic=["Analysis step had token constraints"],
                        code="partial_completion",
                    ),
                )
        else:
            print(f"Error in SGR step: {e}")
            raise


async def process_trading_request(task: str, max_steps: int = 10) -> None:
    """Process trading request with step-by-step visualization."""

    conversation_log = []

    # Save user message to memory
    await save_message_to_memory(task, "user", {"task_type": "trading_analysis"})

    # Initial message
    start_msg = (
        f"**Начинаю финансовый анализ:** {task}\n\n_Максимум шагов: {max_steps}_"
    )
    await cl.Message(
        author="🚀 Старт",
        content=start_msg,
    ).send()

    # Save start message to memory
    await save_message_to_memory(
        start_msg, "system", {"message_type": "analysis_start"}
    )

    for step_num in range(1, max_steps + 1):
        try:
            # Get SGR response
            sgr_response = await run_sgr_step(task, conversation_log)

            # Display reasoning
            await display_sgr_response(sgr_response, step_num)

            # Check if task is completed
            if isinstance(sgr_response.function, ReportTaskCompletion):
                # Execute final report
                result = await execute_tool(
                    "report_completion", sgr_response.function.model_dump()
                )

                # Add to conversation log
                conversation_log.append(
                    {
                        "role": "assistant",
                        "content": sgr_response.plan_remaining_steps_brief[0]
                        if sgr_response.plan_remaining_steps_brief
                        else "Завершение анализа",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"step_{step_num}",
                                "function": {
                                    "name": "report_completion",
                                    "arguments": sgr_response.function.model_dump_json(),
                                },
                            }
                        ],
                    }
                )

                conversation_log.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False),
                        "tool_call_id": f"step_{step_num}",
                    }
                )

                break

            # Execute tool
            tool_name = sgr_response.function.tool
            tool_params = sgr_response.function.model_dump()

            result = dispatch(sgr_response.function)

            # Display result
            await display_tool_result(tool_name, result)

            # Add to conversation log
            conversation_log.append(
                {
                    "role": "assistant",
                    "content": sgr_response.plan_remaining_steps_brief[0]
                    if sgr_response.plan_remaining_steps_brief
                    else "Обработка...",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": f"step_{step_num}",
                            "function": {
                                "name": tool_name,
                                "arguments": sgr_response.function.model_dump_json(),
                            },
                        }
                    ],
                }
            )

            # Add tool result
            result_text = (
                json.dumps(result, ensure_ascii=False, default=str)
                if not isinstance(result, str)
                else result
            )
            conversation_log.append(
                {
                    "role": "tool",
                    "content": result_text,
                    "tool_call_id": f"step_{step_num}",
                }
            )

            # Check if should continue
            if sgr_response.task_completed:
                break

        except Exception as e:
            error_msg = cl.Message(
                author="❌ Ошибка",
                content=f"Ошибка на шаге {step_num}: {str(e)}",
            )
            await error_msg.send()
            break

    # Final message
    final_msg = f"**Анализ завершён за {step_num} шагов**"
    await cl.Message(
        author="📊 Итог",
        content=final_msg,
    ).send()

    # Save final message to memory
    await save_message_to_memory(
        final_msg,
        "system",
        {
            "message_type": "analysis_complete",
            "steps_completed": step_num,
            "task": task,
        },
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def format_tool_name(tool: str) -> str:
    """Format tool name to readable format."""
    replacements = {
        "get_market_data": "Рыночные данные",
        "analyze_trading_opportunity": "Анализ торговых возможностей",
        "generate_forecast": "Генерация прогноза",
        "assess_risk": "Оценка рисков",
        "analyze_news": "Анализ новостей",
        "run_backtest": "Бэктестирование",
        "analyze_web_content": "Анализ веб-контента",
        "research_financial_topic": "Исследование финансовых тем",
        "generate_alphas": "Генерация альфа-факторов",
        "analyze_calibration": "Анализ калибровки прогнозов",
        "manage_risk_controls": "Управление рисками",
        "calculate_enhanced_metrics": "Расчет продвинутых метрик",
        "report_completion": "Финальный отчёт",
    }

    return replacements.get(tool, tool.replace("_", " ").title())


# =============================================================================
# CHAINLIT EVENT HANDLERS
# =============================================================================


async def display_chat_history_summary() -> None:
    """Display chat history summary from memory"""
    try:
        history = get_chat_history(limit=10)
        if history["success"] and history["total_messages"] > 0:
            sessions_count = history["total_sessions"]
            messages_count = history["total_messages"]

            history_msg = f"📚 **История чатов в памяти:**\n\n"
            history_msg += f"• Всего сессий: {sessions_count}\n"
            history_msg += f"• Всего сообщений: {messages_count}\n\n"

            if history["messages"]:
                history_msg += "**Последние сообщения:**\n"
                for msg in history["messages"][-3:]:  # Last 3 messages
                    msg_type = (
                        "👤"
                        if msg["message_type"] == "user"
                        else "🤖"
                        if msg["message_type"] == "assistant"
                        else "🔧"
                    )
                    content_preview = (
                        msg["content"][:100] + "..."
                        if len(msg["content"]) > 100
                        else msg["content"]
                    )
                    history_msg += f"{msg_type} {content_preview}\n"

            await cl.Message(
                author="🧠 Память",
                content=history_msg,
            ).send()
    except Exception as e:
        # Silent fail - history is not critical
        pass


@cl.on_chat_start
async def start_chat():
    """Chat initialization."""

    # Initialize session
    await ensure_chat_session()

    # Check configuration
    try:
        from settings import validate_required_keys

        is_valid, missing_keys = validate_required_keys()
        if not is_valid:
            config_info = f"❌ Отсутствуют ключи: {', '.join(missing_keys)}"
        else:
            config_info = "✅ Конфигурация Azure OpenAI валидна"
    except Exception as e:
        config_info = f"❌ Ошибка конфигурации: {str(e)}"

    # Welcome message
    welcome_msg = f"""
# 💹 SGR Финансовый Торговый Агент

Добро пожаловать в мультиагентную систему финансового анализа с интерактивными графиками!

**🔍 Возможности агента:**
- 📈 **Интерактивные графики:** свечные диаграммы, объемы, индикаторы риска
- Анализ рыночных данных в реальном времени с визуализацией Plotly
- Технический и фундаментальный анализ
- Вероятностные прогнозы с мультиагентным подходом и диаграммами
- Оценка рисков портфеля (VaR, Sharpe ratio) с интерактивными индикаторами
- Анализ новостных настроений с временными графиками
- Бэктестирование стратегий с графиками производительности
- 🌐 **Веб-скрапинг:** извлечение данных с любых финансовых сайтов
- 🔍 **Комплексное исследование:** анализ множества источников одновременно
- 🧮 **Alpha Factory:** генерация альфа-факторов WorldQuant с тепловыми картами IC
- 📊 **Калибровка прогнозов:** Brier Score, ECE, Reliability Diagrams для оценки точности
- ⚠️ **Управление рисками:** Circuit breakers, kill-switch, автоматические лимиты
- 📈 **Продвинутые метрики:** Calmar ratio, Sortino ratio, Implementation Shortfall
- 🧠 **Постоянная память:** сохранение истории чатов и торговых правил
- Торговые рекомендации с ценами входа/выхода

**📊 Статус системы:**
- {config_info}
- Модель: `{settings.azure_openai_deployment_name}`
- Режим торговли: {'📄 Бумажная' if settings.enable_paper_trading else '💰 Реальная'}
- Лимиты риска: {settings.max_position_size:.1%} позиция, {settings.max_daily_drawdown:.1%} просадка
- Токены: {settings.max_completion_tokens:,} completion, адаптивные: {'✓' if settings.enable_adaptive_tokens else '✗'}

**Примеры запросов:**
- "Проанализируй текущую рыночную ситуацию по S&P 500"
- "Создай прогноз на следующую неделю для AAPL с оценкой риска"
- "Рекомендуй портфель акций с бюджетом $100,000"
- "Проведи бэктест стратегии momentum"
- "Оцени влияние новостей на технологический сектор"
- 🌐 "Проанализируй последний отчет Apple с сайта investor.apple.com"
- 🔍 "Исследуй общественное мнение о искусственном интеллекте в финансах"
- 🧮 "Создай альфа-факторы momentum и reversal для FAANG акций"
- 🧮 "Сгенерируй факторы технического анализа с расчетом информационного коэффициента"
- 📊 "Проанализируй калибровку моих прогнозов за последний месяц"
- ⚠️ "Покажи текущий статус системы управления рисками"
- 📈 "Рассчитай Calmar и Sortino ratio для моей стратегии"

**Просто введите ваш вопрос, и я проведу пошаговый финансовый анализ!**
    """

    await cl.Message(author="🤖 Система", content=welcome_msg).send()

    # Display chat history if available
    await display_chat_history_summary()


@cl.on_message
async def handle_message(message: cl.Message):
    """Handle user message."""

    user_query = message.content.strip()

    if not user_query:
        await cl.Message(
            author="⚠️ Система",
            content="Пожалуйста, введите ваш запрос для финансового анализа.",
        ).send()
        return

    # Process trading request
    try:
        await process_trading_request(user_query)
    except Exception as e:
        error_msg = cl.Message(
            author="❌ Критическая ошибка",
            content=f"Произошла ошибка при обработке запроса: {str(e)}",
        )
        await error_msg.send()


if __name__ == "__main__":
    # Launch Chainlit application
    import chainlit as cl

    cl.run()
