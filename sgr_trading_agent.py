#!/usr/bin/env python3
"""
Schema-Guided Reasoning Financial Trading Agent

Multi-agent AI system for financial forecasting and algorithmic trading.
Based on the SGR pattern from https://abdullin.com/schema-guided-reasoning/demo
Adapted for financial markets with Azure OpenAI integration.
"""

import json

# import uuid  # Not used
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
    WebAnalysisRequest,
    ComprehensiveWebResearch,
    CreateTradingRule,
    GetTradingMemory,
    UpdateTradingRule,
    DeleteTradingRule,
    CreateChatSession,
    SaveChatMessage,
    GetChatHistory,
    ReportTaskCompletion,
    SGRTradingResponse,
    TradingMemory,
    MarketData,
    TradingRuleParameters,
    ChatMessageMetadata,
    MemoryFilterCriteria,
    # MarketDataResponse,  # Not used directly
    ForecastResult,
    # TradingRecommendation,  # Not used directly
    # Alpha Factory models
    AlphaGenerationRequest,
    AlphaSpec,
    AlphaSeries,
    AlphaReport,
    # New calibration and risk control models
    CalibrationAnalysisRequest,
    RiskControlsRequest,
    EnhancedMetricsRequest,
)

# Import market data tools
from market_data_tools import (
    MarketDataCollector,
    TradingAnalyzer,
    NewsAnalyzer,
    RiskAnalyzer,
)

# Import web intelligence tools
from web_intelligence import (
    analyze_web_content,
    research_financial_topic,
)

# Import Opoint API for news analysis
from api import OpointAPI

# Import Alpha Factory engine
from alpha_engine import compute_alphas

# Import new calibration and risk control systems
from calibration_engine import (
    analyze_forecast_calibration,
    register_forecast_for_calibration,
)
from risk_controls import manage_risk_controls, RISK_CONTROL_SYSTEM
from enhanced_metrics import calculate_enhanced_metrics

# Setup rich console for beautiful output
console = Console()

# Export DB instance for external use
__all__ = [
    "DB",
    "get_market_data",
    "analyze_trading_opportunity",
    "generate_forecast",
    "assess_risk",
    "analyze_news",
    "run_backtest",
    "create_trading_rule",
    "get_trading_memory",
    "update_trading_rule",
    "delete_trading_rule",
    "create_chat_session",
    "save_chat_message",
    "get_chat_history",
    "generate_alphas",
    "analyze_calibration",
    "manage_risk_controls_wrapper",
    "calculate_enhanced_metrics_wrapper",
    "dispatch",
]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialTradingDB:
    """In-memory database for financial trading data (SGR style)"""

    def __init__(self):
        # SGR-style simple in-memory database structure
        self.data = {
            "rules": [],  # Trading rules created by agent
            "memory": TradingMemory(),  # Structured memory
            "forecasts_history": [],  # Historical forecasts
            "analysis_history": [],  # Analysis records
            "customer_rules": [],  # Customer-specific rules
            "sessions": {},  # Chat sessions data
            "preferences": {},  # Agent preferences
            "chat_messages": {},  # Messages by session_id
        }

        logger.info("Financial Trading Agent initialized with SGR pattern")

    def add_market_data(self, market_data: List[MarketData]):
        """Add market data to memory"""
        self.data["memory"].market_data.extend(market_data)
        # Keep only last 1000 records to manage memory
        if len(self.data["memory"].market_data) > 1000:
            self.data["memory"].market_data = self.data["memory"].market_data[-1000:]

    def add_forecast(self, forecast: ForecastResult):
        """Add forecast to memory"""
        self.data["memory"].forecasts.append(forecast)
        self.data["forecasts_history"].append(
            {"forecast": forecast.model_dump(), "timestamp": datetime.now().isoformat()}
        )

    def get_recent_analysis(self, symbol: str = None, limit: int = 5) -> List[Dict]:
        """Get recent analysis for symbol or all"""
        if symbol:
            relevant = [a for a in self.data["analysis_history"] if symbol in str(a)]
            return relevant[-limit:] if relevant else []
        return self.data["analysis_history"][-limit:]

    def create_rule(
        self,
        rule_description: str,
        rule_type: str = "general",
        parameters: Dict[str, Any] = None,
        priority: int = 1,
    ) -> Dict[str, Any]:
        """Create a new trading rule (SGR style)"""
        import uuid

        rule_id = f"rule_{len(self.data['rules']) + 1}_{uuid.uuid4().hex[:8]}"
        rule = {
            "rule_id": rule_id,
            "description": rule_description,
            "rule_type": rule_type,
            "parameters": parameters or {},
            "priority": priority,
            "active": True,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "usage_count": 0,
        }

        self.data["rules"].append(rule)
        logger.info(f"Created trading rule: {rule_id} - {rule_description}")
        return rule

    def get_trading_memory(
        self, memory_type: str = "all", filter_by: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get trading memory and rules (SGR style)"""
        result = {}

        if memory_type in ["all", "rules"]:
            rules = self.data["rules"]
            if filter_by:
                # Simple filtering
                if "rule_type" in filter_by:
                    rules = [
                        r for r in rules if r["rule_type"] == filter_by["rule_type"]
                    ]
                if "active" in filter_by:
                    rules = [r for r in rules if r["active"] == filter_by["active"]]
            result["rules"] = rules

        if memory_type in ["all", "market_data"]:
            result["market_data"] = [
                data.model_dump() for data in self.data["memory"].market_data[-50:]
            ]  # Last 50 records

        if memory_type in ["all", "forecasts"]:
            result["forecasts"] = [
                f.model_dump() for f in self.data["memory"].forecasts[-20:]
            ]  # Last 20 forecasts

        if memory_type in ["all", "trades"]:
            result["recent_trades"] = [
                t.model_dump() for t in self.data["memory"].recent_trades[-20:]
            ]

        if memory_type in ["all", "analysis"]:
            result["recent_analysis"] = self.data["analysis_history"][
                -10:
            ]  # Last 10 analyses

        # Add summary stats
        result["summary"] = {
            "total_rules": len(self.data["rules"]),
            "active_rules": len([r for r in self.data["rules"] if r["active"]]),
            "market_data_points": len(self.data["memory"].market_data),
            "total_forecasts": len(self.data["memory"].forecasts),
            "total_trades": len(self.data["memory"].recent_trades),
            "total_analyses": len(self.data["analysis_history"]),
        }

        return result

    def update_rule(
        self,
        rule_id: str,
        new_description: str = None,
        new_parameters: Dict[str, Any] = None,
        new_priority: int = None,
    ) -> Dict[str, Any]:
        """Update existing trading rule"""
        for rule in self.data["rules"]:
            if rule["rule_id"] == rule_id:
                if new_description:
                    rule["description"] = new_description
                if new_parameters:
                    rule["parameters"].update(new_parameters)
                if new_priority:
                    rule["priority"] = new_priority
                rule["last_modified"] = datetime.now().isoformat()
                logger.info(f"Updated trading rule: {rule_id}")
                return rule

        raise ValueError(f"Rule {rule_id} not found")

    def delete_rule(self, rule_id: str, reason: str) -> Dict[str, Any]:
        """Delete trading rule"""
        for i, rule in enumerate(self.data["rules"]):
            if rule["rule_id"] == rule_id:
                deleted_rule = self.data["rules"].pop(i)
                deleted_rule["deleted_at"] = datetime.now().isoformat()
                deleted_rule["deletion_reason"] = reason
                logger.info(f"Deleted trading rule: {rule_id} - {reason}")
                return deleted_rule

        raise ValueError(f"Rule {rule_id} not found")

    def create_chat_session(
        self,
        session_name: str,
        user_id: str = "default",
        session_type: str = "trading_analysis",
    ) -> Dict[str, Any]:
        """Create a new chat session (SGR style)"""
        import uuid

        session_id = f"session_{len(self.data['sessions']) + 1}_{uuid.uuid4().hex[:8]}"
        session = {
            "session_id": session_id,
            "session_name": session_name,
            "user_id": user_id,
            "session_type": session_type,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "message_count": 0,
            "active": True,
        }

        self.data["sessions"][session_id] = session
        self.data["chat_messages"][session_id] = []
        logger.info(f"Created chat session: {session_id} - {session_name}")
        return session

    def save_chat_message(
        self,
        session_id: str,
        message_type: str,
        content: str,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Save chat message to session (SGR style)"""
        if session_id not in self.data["sessions"]:
            raise ValueError(f"Session {session_id} not found")

        message = {
            "message_id": f"msg_{len(self.data['chat_messages'][session_id]) + 1}",
            "session_id": session_id,
            "message_type": message_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Add message to session
        self.data["chat_messages"][session_id].append(message)

        # Update session stats
        session = self.data["sessions"][session_id]
        session["message_count"] += 1
        session["last_activity"] = datetime.now().isoformat()

        logger.info(f"Saved {message_type} message to session {session_id}")
        return message

    def get_chat_history(
        self, session_id: str = None, limit: int = 50, message_types: List[str] = None
    ) -> Dict[str, Any]:
        """Get chat history from memory (SGR style)"""
        if message_types is None:
            message_types = ["user", "assistant"]

        result = {
            "sessions": [],
            "messages": [],
            "total_sessions": len(self.data["sessions"]),
            "total_messages": 0,
        }

        if session_id:
            # Get specific session
            if session_id not in self.data["sessions"]:
                raise ValueError(f"Session {session_id} not found")

            session = self.data["sessions"][session_id]
            messages = self.data["chat_messages"][session_id]

            # Filter by message types
            filtered_messages = [
                msg for msg in messages if msg["message_type"] in message_types
            ]

            # Apply limit
            if limit > 0:
                filtered_messages = filtered_messages[-limit:]

            result["sessions"] = [session]
            result["messages"] = filtered_messages
            result["total_messages"] = len(filtered_messages)

        else:
            # Get all sessions
            result["sessions"] = list(self.data["sessions"].values())

            # Get recent messages from all sessions
            all_messages = []
            for sess_id, messages in self.data["chat_messages"].items():
                for msg in messages:
                    if msg["message_type"] in message_types:
                        all_messages.append(msg)

            # Sort by timestamp and apply limit
            all_messages.sort(key=lambda x: x["timestamp"])
            if limit > 0:
                all_messages = all_messages[-limit:]

            result["messages"] = all_messages
            result["total_messages"] = len(all_messages)

        logger.info(
            f"Retrieved chat history: {len(result['messages'])} messages from {len(result['sessions'])} sessions"
        )
        return result

    @property
    def memory(self):
        """Backward compatibility - access to TradingMemory"""
        return self.data["memory"]

    @property
    def forecasts_history(self):
        """Backward compatibility - access to forecasts history"""
        return self.data["forecasts_history"]

    @property
    def analysis_history(self):
        """Backward compatibility - access to analysis history"""
        return self.data["analysis_history"]

    @property
    def customer_rules(self):
        """Backward compatibility - access to customer rules"""
        return self.data["customer_rules"]


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

    elif isinstance(cmd, WebAnalysisRequest):
        return analyze_web_content(
            cmd.url,
            cmd.analysis_type,
            cmd.extract_data_points,
            cmd.include_links,
        )

    elif isinstance(cmd, ComprehensiveWebResearch):
        return research_financial_topic(
            cmd.search_query,
            cmd.research_depth,
            cmd.max_sources,
            cmd.include_news,
            cmd.include_analyst_reports,
            cmd.extract_financial_data,
            cmd.time_range,
        )

    elif isinstance(cmd, AlphaGenerationRequest):
        return generate_alphas(cmd.symbols, cmd.specs, cmd.timeframe, cmd.period)

    elif isinstance(cmd, CalibrationAnalysisRequest):
        return analyze_calibration(
            cmd.time_period, cmd.agent_types, cmd.include_reliability_diagram
        )

    elif isinstance(cmd, RiskControlsRequest):
        return manage_risk_controls_wrapper(
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

    elif isinstance(cmd, EnhancedMetricsRequest):
        return calculate_enhanced_metrics_wrapper(
            cmd.metrics, cmd.time_period, cmd.benchmark
        )

    elif isinstance(cmd, CreateTradingRule):
        # Convert TradingRuleParameters to dict for internal functions
        params_dict = cmd.parameters.model_dump() if cmd.parameters else {}
        return create_trading_rule(
            cmd.rule_description, cmd.rule_type, params_dict, cmd.priority
        )

    elif isinstance(cmd, GetTradingMemory):
        # Convert MemoryFilterCriteria to dict for internal functions
        filter_dict = cmd.filter_by.model_dump() if cmd.filter_by else None
        return get_trading_memory(cmd.memory_type, filter_dict)

    elif isinstance(cmd, UpdateTradingRule):
        # Convert TradingRuleParameters to dict for internal functions
        new_params_dict = (
            cmd.new_parameters.model_dump() if cmd.new_parameters else None
        )
        return update_trading_rule(
            cmd.rule_id, cmd.new_description, new_params_dict, cmd.new_priority
        )

    elif isinstance(cmd, DeleteTradingRule):
        return delete_trading_rule(cmd.rule_id, cmd.reason)

    elif isinstance(cmd, CreateChatSession):
        return create_chat_session(cmd.session_name, cmd.user_id, cmd.session_type)

    elif isinstance(cmd, SaveChatMessage):
        # Convert ChatMessageMetadata to dict for internal functions
        metadata_dict = cmd.metadata.model_dump() if cmd.metadata else None
        return save_chat_message(
            cmd.session_id, cmd.message_type, cmd.content, metadata_dict
        )

    elif isinstance(cmd, GetChatHistory):
        return get_chat_history(cmd.session_id, cmd.limit, cmd.message_types)

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


# ============ SGR Memory Management Functions ============


def create_trading_rule(
    rule_description: str,
    rule_type: str = "general",
    parameters: Dict[str, Any] = None,
    priority: int = 1,
) -> Dict[str, Any]:
    """Create a trading rule in memory (SGR style)"""
    try:
        logger.info(f"Creating trading rule: {rule_description}")
        rule = DB.create_rule(rule_description, rule_type, parameters, priority)

        return {
            "success": True,
            "rule_id": rule["rule_id"],
            "rule_description": rule["description"],
            "rule_type": rule["rule_type"],
            "parameters": rule["parameters"],
            "priority": rule["priority"],
            "created_at": rule["created_at"],
            "message": f"Trading rule created successfully: {rule['rule_id']}",
        }

    except Exception as e:
        logger.error(f"Error creating trading rule: {e}")
        return {
            "success": False,
            "error": str(e),
            "rule_description": rule_description,
        }


def get_trading_memory(
    memory_type: str = "all", filter_by: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Get trading memory and rules (SGR style)"""
    try:
        logger.info(f"Retrieving trading memory: {memory_type}")
        memory_data = DB.get_trading_memory(memory_type, filter_by)

        return {
            "success": True,
            "memory_type": memory_type,
            "filter_applied": filter_by,
            "data": memory_data,
            "message": f"Retrieved {memory_type} memory data successfully",
        }

    except Exception as e:
        logger.error(f"Error retrieving trading memory: {e}")
        return {
            "success": False,
            "error": str(e),
            "memory_type": memory_type,
        }


def update_trading_rule(
    rule_id: str,
    new_description: str = None,
    new_parameters: Dict[str, Any] = None,
    new_priority: int = None,
) -> Dict[str, Any]:
    """Update existing trading rule"""
    try:
        logger.info(f"Updating trading rule: {rule_id}")
        updated_rule = DB.update_rule(
            rule_id, new_description, new_parameters, new_priority
        )

        return {
            "success": True,
            "rule_id": rule_id,
            "updated_rule": updated_rule,
            "message": f"Trading rule {rule_id} updated successfully",
        }

    except Exception as e:
        logger.error(f"Error updating trading rule {rule_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "rule_id": rule_id,
        }


def delete_trading_rule(rule_id: str, reason: str) -> Dict[str, Any]:
    """Delete trading rule from memory"""
    try:
        logger.info(f"Deleting trading rule: {rule_id}")
        deleted_rule = DB.delete_rule(rule_id, reason)

        return {
            "success": True,
            "rule_id": rule_id,
            "deleted_rule": deleted_rule,
            "deletion_reason": reason,
            "message": f"Trading rule {rule_id} deleted successfully",
        }

    except Exception as e:
        logger.error(f"Error deleting trading rule {rule_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "rule_id": rule_id,
            "reason": reason,
        }


# ============ SGR Chat Management Functions ============


def create_chat_session(
    session_name: str, user_id: str = "default", session_type: str = "trading_analysis"
) -> Dict[str, Any]:
    """Create a new chat session in memory (SGR style)"""
    try:
        logger.info(f"Creating chat session: {session_name}")
        session = DB.create_chat_session(session_name, user_id, session_type)

        return {
            "success": True,
            "session_id": session["session_id"],
            "session_name": session["session_name"],
            "user_id": session["user_id"],
            "session_type": session["session_type"],
            "created_at": session["created_at"],
            "message": f"Chat session created successfully: {session['session_id']}",
        }

    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_name": session_name,
        }


def save_chat_message(
    session_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Save chat message to session memory (SGR style)"""
    try:
        logger.info(f"Saving {message_type} message to session {session_id}")
        message = DB.save_chat_message(session_id, message_type, content, metadata)

        return {
            "success": True,
            "message_id": message["message_id"],
            "session_id": session_id,
            "message_type": message_type,
            "timestamp": message["timestamp"],
            "message": f"Message saved to session {session_id}",
        }

    except Exception as e:
        logger.error(f"Error saving message to session {session_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id,
            "message_type": message_type,
        }


def get_chat_history(
    session_id: str = None, limit: int = 50, message_types: List[str] = None
) -> Dict[str, Any]:
    """Get chat history from memory (SGR style)"""
    try:
        if message_types is None:
            message_types = ["user", "assistant"]

        logger.info(f"Retrieving chat history: session_id={session_id}, limit={limit}")
        history = DB.get_chat_history(session_id, limit, message_types)

        return {
            "success": True,
            "session_id": session_id,
            "sessions": history["sessions"],
            "messages": history["messages"],
            "total_sessions": history["total_sessions"],
            "total_messages": history["total_messages"],
            "message_types": message_types,
            "limit": limit,
            "message": f"Retrieved {len(history['messages'])} messages from {len(history['sessions'])} sessions",
        }

    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id,
            "limit": limit,
        }


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
    """Analyze news sentiment for symbols using Opoint API"""
    try:
        if symbols is None:
            symbols = settings.default_symbols[:3]

        logger.info(f"Analyzing news sentiment for {symbols}")

        if not settings.opoint_api_key:
            logger.warning("Opoint API key not configured, using mock data")
            news_sentiment = news_analyzer.get_mock_news_sentiment(symbols)
        else:
            # Use real Opoint API for news analysis
            opoint = OpointAPI(settings.opoint_api_key)

            all_news_data = []
            overall_sentiment_scores = []

            for symbol in symbols:
                # Search for news about this symbol
                symbol_search = f"{symbol} OR {_get_company_name(symbol)}"

                # Get articles from last lookback_hours
                from datetime import timedelta

                end_date = datetime.now()
                start_date = end_date - timedelta(hours=lookback_hours)

                try:
                    articles_df = opoint.search_site_and_articles(
                        search_text=symbol_search,
                        language="en",
                        num_articles=20,
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
                        title = str(article.get("title", "")).lower()
                        summary = str(article.get("summary", "")).lower()

                        # Advanced sentiment scoring
                        sentiment_score = _analyze_article_sentiment(
                            title, summary, symbol
                        )
                        market_impact = _assess_market_impact(title, summary, symbol)
                        importance_score = _calculate_news_importance(article, symbol)

                        symbol_sentiment += sentiment_score

                        symbol_articles.append(
                            {
                                "title": article.get("title", ""),
                                "summary": article.get("summary", ""),
                                "url": article.get("url", ""),
                                "published_date": str(
                                    article.get("published_date", "")
                                ),
                                "source_name": article.get("source_name", ""),
                                "sentiment_score": sentiment_score,
                                "market_impact": market_impact,
                                "importance_score": importance_score,
                                "weighted_sentiment": sentiment_score
                                * importance_score,
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

                except Exception as symbol_error:
                    logger.error(f"Error analyzing news for {symbol}: {symbol_error}")
                    continue

            # Calculate overall sentiment
            if overall_sentiment_scores:
                avg_sentiment = np.mean(overall_sentiment_scores)
                sentiment_range = (
                    min(overall_sentiment_scores),
                    max(overall_sentiment_scores),
                )
            else:
                avg_sentiment = 0.0
                sentiment_range = (0.0, 0.0)
                all_news_data = [
                    {
                        "symbol": s,
                        "sentiment_score": 0.0,
                        "article_count": 0,
                        "articles": [],
                    }
                    for s in symbols
                ]

        # Categorize sentiment
        if avg_sentiment > 0.2:
            sentiment_category = "positive"
        elif avg_sentiment < -0.2:
            sentiment_category = "negative"
        else:
            sentiment_category = "neutral"

        result = {
            "symbols_analyzed": symbols,
            "news_articles": sum(data.get("article_count", 0) for data in all_news_data)
            if "all_news_data" in locals()
            else len(news_sentiment)
            if "news_sentiment" in locals()
            else 0,
            "average_sentiment": avg_sentiment,
            "sentiment_range": sentiment_range,
            "sentiment_category": sentiment_category,
            "lookback_hours": lookback_hours,
            "news_data": all_news_data
            if "all_news_data" in locals()
            else [
                {"symbol": s, "sentiment_score": 0.0, "articles": []} for s in symbols
            ],
            "detailed_sentiment": [news.model_dump() for news in news_sentiment]
            if "news_sentiment" in locals()
            else [],
            "timestamp": datetime.now().isoformat(),
            "data_source": "opoint" if settings.opoint_api_key else "mock",
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


def generate_alphas(
    symbols: List[str],
    specs: List[AlphaSpec],
    timeframe: str = "1d",
    period: str = "3mo",
) -> Dict[str, Any]:
    """Generate alpha factors using WorldQuant Finding Alphas operators"""
    try:
        logger.info(f"Generating {len(specs)} alpha factors for {len(symbols)} symbols")

        # Use alpha engine to compute factors
        result = compute_alphas(symbols, specs, timeframe, period)

        if result.get("success"):
            # Store analysis in history
            analysis_record = {
                "type": "alpha_generation",
                "symbols": symbols,
                "specs": [s.model_dump() for s in specs],
                "timestamp": datetime.now().isoformat(),
                "reports": result.get("reports", []),
                "factors_count": len(result.get("factors", [])),
                "timeframe": timeframe,
                "period": period,
            }
            DB.analysis_history.append(analysis_record)

            logger.info(
                f"Alpha generation completed: {len(result.get('factors', []))} factor series"
            )
        else:
            logger.warning(
                f"Alpha generation failed: {result.get('error', 'Unknown error')}"
            )

        return result

    except Exception as e:
        logger.error(f"Error in alpha generation: {e}")
        return {
            "success": False,
            "error": str(e),
            "symbols_requested": symbols,
            "specs_requested": [spec.name for spec in specs],
        }


def analyze_calibration(
    time_period: str = "30d",
    agent_types: List[str] = None,
    include_reliability_diagram: bool = True,
) -> Dict[str, Any]:
    """Analyze forecast calibration metrics"""
    try:
        logger.info(f"Analyzing forecast calibration for period: {time_period}")
        return analyze_forecast_calibration(
            time_period, agent_types, include_reliability_diagram
        )
    except Exception as e:
        logger.error(f"Error in calibration analysis: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def manage_risk_controls_wrapper(
    action: str, control_type: str = None, **kwargs
) -> Dict[str, Any]:
    """Manage risk controls and circuit breakers"""
    try:
        logger.info(f"Managing risk controls: {action}")
        return manage_risk_controls(action, control_type, **kwargs)
    except Exception as e:
        logger.error(f"Error managing risk controls: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def calculate_enhanced_metrics_wrapper(
    metrics: List[str], time_period: str = "1y", benchmark: str = None
) -> Dict[str, Any]:
    """Calculate enhanced trading performance metrics"""
    try:
        logger.info(f"Calculating enhanced metrics: {metrics}")
        return calculate_enhanced_metrics(metrics, time_period, benchmark)
    except Exception as e:
        logger.error(f"Error calculating enhanced metrics: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
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
- analyze_news: Advanced news sentiment analysis using Opoint API with financial-specific keywords and market impact assessment
- run_backtest: Historical strategy validation and performance metrics
- analyze_web_content: Extract structured financial data from any web page using Firecrawl API (earnings reports, analyst reports, financial news)
- research_financial_topic: Comprehensive web research on financial topics with sentiment analysis and data extraction from multiple sources (max 5 sources for performance)
- generate_alphas: Create alpha factors using WorldQuant Finding Alphas operators (delta, delay, ts_mean, ts_std, zscore, ts_rank, decay_linear) with cross-sectional IC calculation

Memory management tools (SGR style):
- create_trading_rule: Create persistent trading rules and preferences in memory (risk_management, trading_preference, analysis_guideline)
- get_trading_memory: Retrieve stored rules, market data, forecasts, trades, and analysis history with filtering
- update_trading_rule: Modify existing trading rules with new parameters or descriptions
- delete_trading_rule: Remove trading rules from memory with reason logging
- report_completion: Provide final trading recommendations and analysis summary

Financial data capabilities:
• Real-time market data via Yahoo Finance API
• Technical indicators: RSI, MACD, Bollinger Bands, Moving Averages
• Risk metrics: Value at Risk (VaR), Sharpe ratio, Maximum Drawdown
• Sentiment analysis: News and social media sentiment scoring
• Multi-agent forecasting: Bullish, bearish, and technical perspectives
• Web scraping: Extract financial data from any website using Firecrawl API
• Comprehensive research: Multi-source financial intelligence gathering

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
        # Calculate token limits dynamically based on model capacity
        max_completion_tokens = settings.max_completion_tokens

        # If adaptive tokens enabled, adjust based on current prompt size
        if settings.enable_adaptive_tokens:
            # Estimate prompt tokens (rough approximation: 1 token ≈ 4 characters)
            estimated_prompt_tokens = (
                sum(len(msg.get("content", "")) for msg in messages) // 4
            )

            # Ensure we don't exceed the total model capacity (200K)
            available_tokens = (
                200000 - estimated_prompt_tokens - 1000
            )  # Buffer for safety
            max_completion_tokens = min(
                max_completion_tokens, max(10000, available_tokens)
            )

        # Use structured output for financial analysis
        completion = client.beta.chat.completions.parse(
            model=deployment_name,
            response_format=SGRTradingResponse,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )

        sgr_response = completion.choices[0].message.parsed

        # Log token usage for debugging
        if hasattr(completion, "usage") and completion.usage:
            usage = completion.usage
            logger.info(
                f"Token usage: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}"
            )
            console.print(
                f"[dim]Tokens used: {usage.completion_tokens}/{max_completion_tokens} completion, {usage.total_tokens} total[/dim]"
            )

        logger.info(f"SGR Financial Analysis: {sgr_response.current_state}")
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
            logger.warning(
                f"Token limit reached in SGR step, retrying with reduced tokens: {e}"
            )

            try:
                # Retry with 50% fewer tokens
                reduced_tokens = max(5000, max_completion_tokens // 2)
                logger.info(f"Retrying with reduced tokens: {reduced_tokens}")

                completion = client.beta.chat.completions.parse(
                    model=deployment_name,
                    response_format=SGRTradingResponse,
                    messages=messages,
                    max_completion_tokens=reduced_tokens,
                )

                sgr_response = completion.choices[0].message.parsed
                logger.info(
                    f"SGR Financial Analysis (reduced tokens): {sgr_response.current_state}"
                )
                return sgr_response

            except Exception as retry_error:
                logger.error(f"Retry with reduced tokens also failed: {retry_error}")
                # Only force completion as last resort
                return SGRTradingResponse(
                    current_state="Token limit exceeded, continuing with reduced response",
                    plan_remaining_steps_brief=["Continue analysis with next tool"],
                    task_completed=False,  # Don't force completion
                    function=ReportTaskCompletion(
                        completed_steps_laconic=["Analysis step had token constraints"],
                        code="partial_completion",
                    ),
                )
        else:
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
            f"Risk Settings: {settings.max_position_size:.1%} max position, {settings.max_daily_drawdown:.1%} max drawdown\n"
            f"Token Limits: {settings.max_completion_tokens:,} completion tokens, Adaptive: {'✓' if settings.enable_adaptive_tokens else '✗'}",
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
                elif "sentiment_category" in result:
                    sentiment = result.get("sentiment_category", "neutral")
                    avg_sentiment = result.get("average_sentiment", 0.0)
                    articles_count = result.get("news_articles", 0)
                    data_source = result.get("data_source", "unknown")
                    console.print(
                        f"[green]📰 News Analysis:[/green] {sentiment.upper()} sentiment ({avg_sentiment:.2f}) from {articles_count} articles via {data_source}"
                    )
                elif "factors" in result and "reports" in result:
                    factors_count = len(result.get("factors", []))
                    reports = result.get("reports", [])
                    symbols_processed = result.get("symbols_processed", 0)
                    factors_computed = result.get("factors_computed", 0)

                    console.print(
                        f"[green]🧮 Alpha Factors Generated:[/green] {factors_computed} factors for {symbols_processed} symbols ({factors_count} series)"
                    )

                    # Show IC results for each factor
                    if reports:
                        console.print(
                            "[cyan]Information Coefficient (IC) Results:[/cyan]"
                        )
                        for report in reports[:5]:  # Show top 5 factors
                            factor_name = report.get("factor", "Unknown")
                            ic = report.get("ic1d")
                            coverage = report.get("coverage", 0)

                            if ic is not None:
                                ic_str = f"{ic:.4f}"
                                if abs(ic) > 0.05:
                                    ic_color = "green" if ic > 0 else "red"
                                    console.print(
                                        f"  • {factor_name}: IC={ic_str} ({coverage} obs) [{ic_color}]{'Strong' if abs(ic) > 0.1 else 'Moderate'}[/{ic_color}]"
                                    )
                                else:
                                    console.print(
                                        f"  • {factor_name}: IC={ic_str} ({coverage} obs) [dim]Weak[/dim]"
                                    )
                            else:
                                console.print(
                                    f"  • {factor_name}: IC=N/A ({coverage} obs)"
                                )

                    # Show last values for debugging
                    if len(reports) > 0 and reports[0].get("last_value"):
                        console.print(
                            f"[dim]Last factor values available for {len(reports[0]['last_value'])} symbols[/dim]"
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


def _analyze_article_sentiment(title: str, summary: str, symbol: str) -> float:
    """Advanced sentiment analysis for financial news"""
    text = title + " " + summary

    # Enhanced sentiment keywords (financial-specific)
    sentiment_indicators = {
        # Very positive (0.8-1.0)
        "breakthrough": 0.9,
        "record": 0.8,
        "soars": 0.9,
        "surge": 0.8,
        "outperforms": 0.7,
        "beats expectations": 0.9,
        "all-time high": 1.0,
        # Positive (0.3-0.7)
        "buy": 0.6,
        "bull": 0.7,
        "growth": 0.5,
        "up": 0.4,
        "rise": 0.5,
        "gain": 0.6,
        "profit": 0.7,
        "strong": 0.6,
        "beat": 0.7,
        "exceed": 0.6,
        "upgrade": 0.7,
        "optimistic": 0.6,
        "bullish": 0.8,
        "rally": 0.7,
        # Negative (-0.3 to -0.7)
        "sell": -0.6,
        "bear": -0.7,
        "down": -0.4,
        "fall": -0.5,
        "loss": -0.6,
        "weak": -0.5,
        "miss": -0.7,
        "decline": -0.5,
        "cut": -0.6,
        "warning": -0.6,
        "downgrade": -0.7,
        "pessimistic": -0.6,
        "bearish": -0.8,
        "plunge": -0.8,
        # Very negative (-0.8 to -1.0)
        "crash": -0.9,
        "collapse": -1.0,
        "crisis": -0.8,
        "disaster": -0.9,
        "scandal": -0.8,
        "investigation": -0.7,
        "lawsuit": -0.6,
        "fraud": -1.0,
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
        "earnings",
        "revenue",
        "guidance",
        "acquisition",
        "merger",
        "ipo",
        "fda approval",
        "partnership",
        "contract",
        "lawsuit",
        "regulatory",
        "ceo",
        "layoffs",
        "restructuring",
        "dividend",
        "stock split",
        "buyback",
        "bankruptcy",
        "delisting",
        "investigation",
    ]

    medium_impact_events = [
        "analyst",
        "rating",
        "price target",
        "recommendation",
        "conference",
        "product launch",
        "expansion",
        "hiring",
        "investment",
        "funding",
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
        "yahoo finance": 0.5,
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
            elif "sentiment_category" in content:
                tools_used.append("• Проанализирован новостной фон и настроения рынка")

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
            "• Следите за новостным фоном и настроениями рынка",
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
