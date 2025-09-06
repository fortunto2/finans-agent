#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Trading SGR Agent - Chainlit GUI Interface
Beautiful web interface for the financial trading agent with step-by-step visualization
"""

import json
from typing import Any, Dict, List, Union
# from datetime import datetime  # Not used directly

import chainlit as cl
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
)

from sgr_trading_agent import (
    get_market_data,
    analyze_trading_opportunity,
    generate_forecast,
    assess_risk,
    analyze_news,
    run_backtest,
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
    """Display market data in a beautiful format."""

    content = "**📊 Рыночные данные:**\n\n"

    content += f"**Символы:** {', '.join(data.get('symbols_analyzed', []))}\n"
    content += f"**Точек данных:** {data.get('data_points', 0)}\n"
    content += f"**Рыночный тренд:** {data.get('market_trend', 'unknown')}\n"
    content += f"**Волатильность:** {data.get('volatility_assessment', 'unknown')}\n\n"

    if data.get("key_insights"):
        content += "**🔍 Ключевые инсайты:**\n"
        for insight in data["key_insights"]:
            content += f"• {insight}\n"

    msg = cl.Message(
        author="📈 Рыночные данные",
        content=content,
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
    """Display forecast results with probabilities."""

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

    msg = cl.Message(
        author="🔮 Прогноз",
        content=content,
    )
    await msg.send()


async def display_risk_assessment(risk: Dict[str, Any]) -> None:
    """Display risk assessment results."""

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

    msg = cl.Message(
        author="⚠️ Оценка рисков",
        content=content,
    )
    await msg.send()


async def display_news_sentiment(news: Dict[str, Any]) -> None:
    """Display news sentiment analysis."""

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

    msg = cl.Message(
        author="📰 Новости",
        content=content,
    )
    await msg.send()


async def display_backtest_result(backtest: Dict[str, Any]) -> None:
    """Display backtest results."""

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

    msg = cl.Message(
        author="🔄 Бэктест",
        content=content,
    )
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
    """Display final answer and recommendations."""

    content = "**🎯 ИТОГОВЫЙ АНАЛИЗ:**\n\n"

    if completion.final_recommendation:
        content += (
            f"**💡 Торговая рекомендация:**\n{completion.final_recommendation}\n\n"
        )

    content += "**✅ Выполненные шаги:**\n"
    for i, step in enumerate(completion.completed_steps_laconic, 1):
        content += f"{i}. {step}\n"

    # Show portfolio status if available
    if DB.memory.positions:
        content += "\n**💼 Текущие позиции:**\n"
        for position in DB.memory.positions[-5:]:
            content += f"• {position.symbol}: {position.quantity} акций @ ${position.avg_cost:.2f}\n"

    msg = cl.Message(
        author="🏁 Завершение",
        content=content,
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

    # Use structured output
    completion = await client.beta.chat.completions.parse(
        model=deployment_name,
        response_format=SGRTradingResponse,
        messages=messages,
        max_completion_tokens=2000,  # Increased from 1000 to 2000
    )

    return completion.choices[0].message.parsed


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

Добро пожаловать в мультиагентную систему финансового анализа!

**🔍 Возможности агента:**
- Анализ рыночных данных в реальном времени
- Технический и фундаментальный анализ
- Вероятностные прогнозы с мультиагентным подходом
- Оценка рисков портфеля (VaR, Sharpe ratio)
- Анализ новостных настроений
- Бэктестирование стратегий
- 🌐 **Веб-скрапинг:** извлечение данных с любых финансовых сайтов
- 🔍 **Комплексное исследование:** анализ множества источников одновременно
- 🧠 **Постоянная память:** сохранение истории чатов и торговых правил
- Торговые рекомендации с ценами входа/выхода

**📊 Статус системы:**
- {config_info}
- Модель: `{settings.azure_openai_deployment_name}`
- Режим торговли: {'📄 Бумажная' if settings.enable_paper_trading else '💰 Реальная'}
- Лимиты риска: {settings.max_position_size:.1%} позиция, {settings.max_daily_drawdown:.1%} просадка

**Примеры запросов:**
- "Проанализируй текущую рыночную ситуацию по S&P 500"
- "Создай прогноз на следующую неделю для AAPL с оценкой риска"
- "Рекомендуй портфель акций с бюджетом $100,000"
- "Проведи бэктест стратегии momentum"
- "Оцени влияние новостей на технологический сектор"
- 🌐 "Проанализируй последний отчет Apple с сайта investor.apple.com"
- 🔍 "Исследуй общественное мнение о искусственном интеллекте в финансах"

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
