#!/usr/bin/env python3
"""
Deep Research SGR Agent for Financial Analysis

Full implementation of Schema-Guided Reasoning with adaptive NextStep planning.
Based on SGR-classic pattern from abdullin.com/schema-guided-reasoning/demo
"""

import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Union, Optional

from openai import AzureOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

# Import our models and settings
from settings import settings
from deep_research_models import (
    NextStep,
    ResearchContext,
    ResearchInsight,
    ResearchFindings,
    ResearchPhase,
    ResearchAction,
    DeepResearchRequest,
    CompetitorAnalysisRequest,
    EconomicIndicatorRequest,
    ValidationRequest,
    SynthesisRequest,
)
from models import (
    MarketDataRequest,
    NewsAnalysisRequest,
    RiskAssessmentRequest,
    TradingAnalysisRequest,
    ForecastRequest,
    ReportTaskCompletion,
)

# Import the deep research engine components
from deep_research_engine import (
    DeepResearchEngine,
    analyze_competitors,
    analyze_economic_indicators,
    validate_research_findings,
    synthesize_research_results,
)

# Setup console and logging
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepResearchResponse(BaseModel):
    """SGR Response for Deep Research Agent"""

    current_state: str = Field(description="Current research state and progress")
    research_phase: ResearchPhase = Field(description="Current phase of research")
    insights_summary: str = Field(description="Summary of insights gathered so far")
    next_step: NextStep = Field(description="Adaptive next step planning")
    function: Union[
        MarketDataRequest,
        NewsAnalysisRequest,
        RiskAssessmentRequest,
        TradingAnalysisRequest,
        ForecastRequest,
        CompetitorAnalysisRequest,
        EconomicIndicatorRequest,
        ValidationRequest,
        SynthesisRequest,
        ReportTaskCompletion,
    ] = Field(description="Next research tool to execute")
    research_depth: int = Field(description="Current research depth level")
    confidence_progression: List[float] = Field(
        description="Confidence scores over time"
    )

    class Config:
        extra = "forbid"


class DeepResearchAgent:
    """Main Deep Research SGR Agent"""

    def __init__(self):
        self.client = self._setup_azure_client()
        self.engine = DeepResearchEngine(self.client)
        self.current_session: Optional[ResearchContext] = None
        self.memory: Dict[str, Any] = {}

    def _setup_azure_client(self) -> AzureOpenAI:
        """Setup Azure OpenAI client"""
        return AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

    def dispatch_tool(self, cmd) -> Dict[str, Any]:
        """Dispatch tool commands to appropriate handlers"""

        if isinstance(cmd, MarketDataRequest):
            return self._execute_market_data(cmd)
        elif isinstance(cmd, NewsAnalysisRequest):
            return self._execute_news_analysis(cmd)
        elif isinstance(cmd, RiskAssessmentRequest):
            return self._execute_risk_assessment(cmd)
        elif isinstance(cmd, TradingAnalysisRequest):
            return self._execute_trading_analysis(cmd)
        elif isinstance(cmd, ForecastRequest):
            return self._execute_forecast(cmd)
        elif isinstance(cmd, CompetitorAnalysisRequest):
            return analyze_competitors(cmd)
        elif isinstance(cmd, EconomicIndicatorRequest):
            return analyze_economic_indicators(cmd)
        elif isinstance(cmd, ValidationRequest):
            return validate_research_findings(cmd)
        elif isinstance(cmd, SynthesisRequest):
            return synthesize_research_results(cmd)
        elif isinstance(cmd, ReportTaskCompletion):
            return self._finalize_research(cmd)
        else:
            return {"error": f"Unknown tool type: {type(cmd)}"}

    def _execute_market_data(self, request: MarketDataRequest) -> Dict[str, Any]:
        """Execute comprehensive market data analysis"""
        from market_data_tools import MarketDataCollector

        collector = MarketDataCollector()
        try:
            result = collector.get_comprehensive_data(
                symbols=request.symbols,
                timeframe=request.timeframe,
                period=request.period,
            )

            # Add research context
            result["research_context"] = {
                "tool": "market_data",
                "depth": "comprehensive",
                "confidence": 0.85,
                "insights_generated": len(result.get("key_insights", [])),
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            logger.error(f"Market data analysis failed: {e}")
            return {"error": str(e), "tool": "market_data"}

    def _execute_news_analysis(self, request: NewsAnalysisRequest) -> Dict[str, Any]:
        """Execute advanced news sentiment analysis"""
        from market_data_tools import NewsAnalyzer

        analyzer = NewsAnalyzer()
        try:
            result = analyzer.analyze_market_sentiment(
                symbols=request.symbols,
                sources=request.sources,
                lookback_hours=request.lookback_hours,
            )

            # Enhanced with research context
            result["research_context"] = {
                "tool": "news_analysis",
                "sentiment_strength": abs(result.get("average_sentiment", 0.0)),
                "confidence": min(0.9, result.get("news_articles", 0) / 20),
                "market_impact": self._assess_news_impact(result),
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            return {"error": str(e), "tool": "news_analysis"}

    def _execute_risk_assessment(
        self, request: RiskAssessmentRequest
    ) -> Dict[str, Any]:
        """Execute comprehensive risk assessment"""
        from market_data_tools import RiskAnalyzer

        analyzer = RiskAnalyzer()
        try:
            result = analyzer.comprehensive_risk_analysis(
                symbol=request.symbol,
                trade_amount=request.trade_amount,
                assessment_type=request.assessment_type,
            )

            # Enhanced risk context
            result["research_context"] = {
                "tool": "risk_assessment",
                "risk_score": self._calculate_risk_score(result),
                "confidence": 0.8,
                "critical_factors": self._identify_critical_risks(result),
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"error": str(e), "tool": "risk_assessment"}

    def _execute_trading_analysis(
        self, request: TradingAnalysisRequest
    ) -> Dict[str, Any]:
        """Execute comprehensive trading analysis"""
        from market_data_tools import TradingAnalyzer

        analyzer = TradingAnalyzer()
        try:
            result = analyzer.analyze_trading_opportunity(
                symbol=request.symbol,
                analysis_type=request.analysis_type,
                budget=request.budget,
                risk_tolerance=request.risk_tolerance,
            )

            # Enhanced trading context
            result["research_context"] = {
                "tool": "trading_analysis",
                "signal_strength": self._assess_signal_strength(result),
                "confidence": 0.75,
                "opportunity_score": self._calculate_opportunity_score(result),
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            logger.error(f"Trading analysis failed: {e}")
            return {"error": str(e), "tool": "trading_analysis"}

    def _execute_forecast(self, request: ForecastRequest) -> Dict[str, Any]:
        """Execute probabilistic forecasting"""
        try:
            # Mock advanced forecasting (in real system would use ML models)
            import numpy as np

            forecast_probability = np.random.uniform(0.3, 0.8)
            confidence_interval = np.random.uniform(0.1, 0.3)

            result = {
                "question": request.question,
                "symbols": request.symbols,
                "forecast_horizon": request.forecast_horizon,
                "probability": forecast_probability,
                "confidence_interval": confidence_interval,
                "agent_consensus": {
                    "bullish_agent": np.random.uniform(0.4, 0.9),
                    "bearish_agent": np.random.uniform(0.1, 0.6),
                    "technical_agent": np.random.uniform(0.3, 0.8),
                    "sentiment_agent": np.random.uniform(0.2, 0.7),
                },
                "forecast_rationale": f"Based on {len(request.agent_types)} agent perspectives",
                "research_context": {
                    "tool": "forecast",
                    "forecast_confidence": forecast_probability,
                    "consensus_strength": 0.7,
                    "temporal_scope": request.forecast_horizon,
                    "timestamp": datetime.now().isoformat(),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Forecast execution failed: {e}")
            return {"error": str(e), "tool": "forecast"}

    def _finalize_research(self, request: ReportTaskCompletion) -> Dict[str, Any]:
        """Finalize research and generate final findings"""
        if not self.current_session:
            return {"error": "No active research session"}

        try:
            # Generate final findings using the engine
            findings = self.engine.generate_final_findings(self.current_session)

            # Create completion report
            completion_report = {
                "research_complete": True,
                "final_findings": findings,
                "session_summary": {
                    "research_id": self.current_session.research_id,
                    "total_insights": len(self.current_session.insights_collected),
                    "final_confidence": self.current_session.confidence_level,
                    "research_depth": self.current_session.research_depth,
                    "phases_completed": list(
                        set(
                            insight.phase.value
                            for insight in self.current_session.insights_collected
                        )
                    ),
                },
                "recommendations": request.recommendations,
                "analysis_summary": request.analysis_summary,
                "timestamp": datetime.now().isoformat(),
            }

            # Archive session
            self.memory[self.current_session.research_id] = {
                "context": self.current_session,
                "findings": findings,
                "completion_report": completion_report,
            }

            logger.info(
                f"Research session {self.current_session.research_id} completed"
            )
            return completion_report

        except Exception as e:
            logger.error(f"Research finalization failed: {e}")
            return {"error": str(e), "research_incomplete": True}

    def run_deep_research_session(self, research_request: DeepResearchRequest) -> None:
        """Run complete deep research session with adaptive planning"""

        console.print(
            Panel.fit(
                f"🔬 [bold blue]Deep Research Session Started[/bold blue]\n"
                f"📊 Research Question: {research_request.research_question}\n"
                f"📈 Symbols: {', '.join(research_request.symbols)}\n"
                f"⚙️ Scope: {research_request.research_scope}\n"
                f"🎯 Max Depth: {research_request.max_depth}",
                title="Deep Research Agent",
                border_style="blue",
            )
        )

        # Initialize research session
        self.current_session = self.engine.start_deep_research(research_request)

        step_count = 0
        max_steps = research_request.max_depth

        while step_count < max_steps:
            step_count += 1

            # Get next adaptive step
            next_step = self.engine.get_next_step(self.current_session)

            console.print(
                f"\n🎯 [bold]Step {step_count}/{max_steps}[/bold] - {next_step.research_priority.upper()} Priority"
            )
            console.print(
                f"🔍 Phase: {self.current_session.current_phase.value.replace('_', ' ').title()}"
            )
            console.print(
                f"⚡ Action: {next_step.next_action.value.replace('_', ' ').title()}"
            )
            console.print(f"🧠 Reasoning: {next_step.reasoning}")

            # Execute SGR reasoning step
            try:
                sgr_response = self._execute_sgr_step(next_step)

                if sgr_response.function:
                    # Execute the selected tool
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            f"Executing {sgr_response.function.__class__.__name__}...",
                            total=None,
                        )

                        tool_result = self.dispatch_tool(sgr_response.function)
                        progress.remove_task(task)

                    # Process result and create insight
                    if "error" not in tool_result:
                        insight = self._create_insight_from_result(
                            tool_result,
                            next_step.next_action,
                            self.current_session.current_phase,
                        )

                        self.engine.add_insight(self.current_session, insight)

                        # Display result
                        self._display_step_result(tool_result, insight)

                        # Update research progress
                        self.current_session.research_depth = step_count

                        # Check if should continue
                        if not self.engine.should_continue_research(
                            self.current_session, next_step
                        ):
                            console.print(
                                f"\n✅ [green]Research criteria satisfied at step {step_count}[/green]"
                            )
                            break

                        # Advance phase if appropriate
                        if next_step.estimated_completion_progress > 0.9:
                            self.engine.advance_phase(self.current_session)

                    else:
                        console.print(
                            f"❌ [red]Tool execution failed: {tool_result.get('error')}[/red]"
                        )

                else:
                    # Research completion requested
                    console.print(
                        "\n🎯 [green]Research completion requested by SGR agent[/green]"
                    )
                    break

            except Exception as e:
                logger.error(f"SGR step execution failed: {e}")
                console.print(f"❌ [red]Step failed: {str(e)}[/red]")
                continue

        # Finalize research
        console.print(
            f"\n🏁 [bold yellow]Finalizing Deep Research Session...[/bold yellow]"
        )

        completion_request = ReportTaskCompletion(
            recommendations=["Deep research completed"],
            analysis_summary=f"Comprehensive analysis of {research_request.symbols}",
            confidence_score=self.current_session.confidence_level,
        )

        final_report = self._finalize_research(completion_request)

        # Display final results
        self._display_final_results(final_report)

    def _execute_sgr_step(self, next_step: NextStep) -> DeepResearchResponse:
        """Execute SGR reasoning to determine next tool"""

        # Prepare context for SGR
        insights_summary = f"Collected {len(self.current_session.insights_collected)} insights across {len(set(i.phase for i in self.current_session.insights_collected))} phases"

        system_prompt = f"""You are a Deep Research Financial Analyst using Schema-Guided Reasoning.

Current Research Context:
- Research Question: {self.current_session.research_question}
- Symbols: {self.current_session.primary_symbols}
- Current Phase: {self.current_session.current_phase.value}
- Research Depth: {self.current_session.research_depth}/{self.current_session.max_depth}
- Confidence Level: {self.current_session.confidence_level:.2f}

Next Step Plan:
- Action: {next_step.next_action.value}
- Reasoning: {next_step.reasoning}
- Priority: {next_step.research_priority}
- Expected Progress: {next_step.estimated_completion_progress:.2f}

Available Tools:
- MarketDataRequest: Comprehensive market data and technical analysis
- NewsAnalysisRequest: Advanced news sentiment and market impact analysis  
- RiskAssessmentRequest: Multi-dimensional risk evaluation
- TradingAnalysisRequest: Trading opportunity assessment with signals
- ForecastRequest: Probabilistic forecasting with multiple agent perspectives
- CompetitorAnalysisRequest: Competitive positioning and peer analysis
- EconomicIndicatorRequest: Macroeconomic factors and correlations
- ValidationRequest: Cross-validation of research findings
- SynthesisRequest: Synthesis of all research into actionable insights
- ReportTaskCompletion: Finalize research and generate recommendations

Your task: Select the most appropriate tool to execute the planned next step. Consider the current research phase, accumulated insights, and knowledge gaps."""

        user_prompt = f"""Given the next step plan, what specific tool should be executed?

Next Action: {next_step.next_action.value}
Current Understanding: {next_step.current_understanding}
Knowledge Gaps: {", ".join(next_step.knowledge_gaps)}

Provide the SGR response with current state, insights summary, and the specific tool to execute."""

        try:
            completion = self.client.beta.chat.completions.parse(
                model=settings.azure_openai_deployment_name,
                response_format=DeepResearchResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=1000,
            )

            response = completion.choices[0].message.parsed
            response.next_step = next_step  # Ensure next_step is included
            response.research_depth = self.current_session.research_depth
            response.confidence_progression = [
                insight.confidence
                for insight in self.current_session.insights_collected
            ]

            return response

        except Exception as e:
            logger.error(f"SGR execution failed: {e}")
            # Fallback response
            return self._create_fallback_response(next_step)

    def _create_fallback_response(self, next_step: NextStep) -> DeepResearchResponse:
        """Create fallback SGR response if LLM fails"""

        # Map actions to tools
        action_to_tool = {
            ResearchAction.COLLECT_MARKET_DATA: MarketDataRequest(
                symbols=self.current_session.primary_symbols,
                timeframe="1d",
                period="3mo",
            ),
            ResearchAction.NEWS_SENTIMENT: NewsAnalysisRequest(
                symbols=self.current_session.primary_symbols,
                sources=["reuters", "bloomberg"],
                lookback_hours=48,
            ),
            ResearchAction.RISK_METRICS: RiskAssessmentRequest(
                symbol=self.current_session.primary_symbols[0],
                assessment_type="comprehensive",
            ),
            ResearchAction.FORECAST_MODELS: ForecastRequest(
                question=f"Price direction for {self.current_session.primary_symbols}",
                symbols=self.current_session.primary_symbols,
                forecast_horizon="1_week",
            ),
            ResearchAction.COMPETITOR_ANALYSIS: CompetitorAnalysisRequest(
                primary_symbol=self.current_session.primary_symbols[0]
            ),
            ResearchAction.ECONOMIC_INDICATORS: EconomicIndicatorRequest(),
            ResearchAction.VALIDATE_FINDINGS: ValidationRequest(
                findings_to_validate=[
                    insight.content
                    for insight in self.current_session.insights_collected[-3:]
                ]
            ),
            ResearchAction.SYNTHESIZE_RESULTS: SynthesisRequest(
                research_context=self.current_session
            ),
        }

        selected_tool = action_to_tool.get(
            next_step.next_action,
            MarketDataRequest(symbols=self.current_session.primary_symbols),
        )

        return DeepResearchResponse(
            current_state=f"Fallback execution of {next_step.next_action.value}",
            research_phase=self.current_session.current_phase,
            insights_summary=f"Collected {len(self.current_session.insights_collected)} insights",
            next_step=next_step,
            function=selected_tool,
            research_depth=self.current_session.research_depth,
            confidence_progression=[
                i.confidence for i in self.current_session.insights_collected
            ],
        )

    def _create_insight_from_result(
        self, result: Dict[str, Any], action: ResearchAction, phase: ResearchPhase
    ) -> ResearchInsight:
        """Create research insight from tool result"""

        # Extract key content based on tool type
        content_parts = []
        confidence = 0.7  # Base confidence
        impact_score = 0.6  # Base impact
        sources = []

        if "research_context" in result:
            research_ctx = result["research_context"]
            confidence = research_ctx.get("confidence", 0.7)
            sources.append(research_ctx.get("tool", "unknown"))

        # Extract meaningful content
        if action == ResearchAction.COLLECT_MARKET_DATA:
            if "market_trend" in result:
                content_parts.append(f"Market trend: {result['market_trend']}")
            if "key_insights" in result:
                content_parts.extend(result["key_insights"][:2])
            impact_score = 0.8

        elif action == ResearchAction.NEWS_SENTIMENT:
            if "sentiment_category" in result:
                sentiment = result["sentiment_category"]
                avg_sentiment = result.get("average_sentiment", 0.0)
                content_parts.append(
                    f"News sentiment: {sentiment} ({avg_sentiment:.2f})"
                )
            if "news_articles" in result:
                content_parts.append(f"Analyzed {result['news_articles']} articles")
            impact_score = 0.7

        elif action == ResearchAction.RISK_METRICS:
            if "risk_level" in result:
                content_parts.append(f"Risk level: {result['risk_level']}")
            if "var_1d" in result:
                content_parts.append(f"1-day VaR: {result['var_1d']:.2%}")
            impact_score = 0.9

        elif action == ResearchAction.COMPETITOR_ANALYSIS:
            if "competitive_advantages" in result:
                content_parts.extend(result["competitive_advantages"][:2])
            impact_score = 0.6

        elif action == ResearchAction.ECONOMIC_INDICATORS:
            if "overall_economic_outlook" in result:
                content_parts.append(
                    f"Economic outlook: {result['overall_economic_outlook']}"
                )
            impact_score = 0.7

        # Fallback content
        if not content_parts:
            content_parts = [f"Executed {action.value.replace('_', ' ')} analysis"]

        content = "; ".join(content_parts)

        return ResearchInsight(
            insight_id=str(uuid.uuid4()),
            phase=phase,
            action=action,
            content=content,
            confidence=confidence,
            impact_score=impact_score,
            sources=sources or ["analysis_engine"],
            timestamp=datetime.now(),
            metadata={"tool_result_keys": list(result.keys())},
        )

    def _display_step_result(
        self, result: Dict[str, Any], insight: ResearchInsight
    ) -> None:
        """Display step execution result"""

        table = Table(
            title="Step Result", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Action", insight.action.value.replace("_", " ").title())
        table.add_row("Confidence", f"{insight.confidence:.1%}")
        table.add_row("Impact Score", f"{insight.impact_score:.1%}")
        table.add_row("Sources", ", ".join(insight.sources))

        console.print(table)
        console.print(f"💡 [yellow]Insight:[/yellow] {insight.content}")

    def _display_final_results(self, final_report: Dict[str, Any]) -> None:
        """Display final research results"""

        if "final_findings" in final_report:
            findings = final_report["final_findings"]

            console.print(
                Panel.fit(
                    f"🎯 [bold green]Research Complete![/bold green]\n\n"
                    f"📊 Investment Thesis: {findings.investment_thesis}\n"
                    f"📈 Recommendation: {findings.recommended_action.upper()}\n"
                    f"💰 Position Sizing: {findings.position_sizing or 'Not specified'}\n"
                    f"🎯 Confidence: {findings.confidence_score:.1%}\n"
                    f"⚠️ Risk Assessment: {findings.risk_assessment}",
                    title="Final Research Findings",
                    border_style="green",
                )
            )

            # Key findings table
            if findings.key_findings:
                findings_table = Table(title="Key Research Findings", show_header=False)
                findings_table.add_column("Finding", style="white")

                for i, finding in enumerate(findings.key_findings[:5], 1):
                    findings_table.add_row(f"{i}. {finding}")

                console.print(findings_table)

            # Strategy recommendations
            if findings.monitoring_plan:
                console.print("\n📋 [bold]Monitoring Plan:[/bold]")
                for item in findings.monitoring_plan:
                    console.print(f"  • {item}")

        # Session summary
        if "session_summary" in final_report:
            summary = final_report["session_summary"]
            console.print(f"\n📈 [blue]Session Summary:[/blue]")
            console.print(f"  Research ID: {summary['research_id']}")
            console.print(f"  Total Insights: {summary['total_insights']}")
            console.print(f"  Final Confidence: {summary['final_confidence']:.1%}")
            console.print(f"  Research Depth: {summary['research_depth']}")
            console.print(
                f"  Phases Completed: {', '.join(summary['phases_completed'])}"
            )

    # Helper methods for result processing
    def _assess_news_impact(self, result: Dict[str, Any]) -> str:
        """Assess market impact of news sentiment"""
        sentiment = result.get("average_sentiment", 0.0)
        articles = result.get("news_articles", 0)

        if abs(sentiment) > 0.3 and articles > 10:
            return "high"
        elif abs(sentiment) > 0.1 and articles > 5:
            return "medium"
        else:
            return "low"

    def _calculate_risk_score(self, result: Dict[str, Any]) -> float:
        """Calculate numeric risk score"""
        risk_level = result.get("risk_level", "medium")
        var_1d = result.get("var_1d", 0.02)

        risk_scores = {"low": 0.3, "medium": 0.6, "high": 0.9}
        base_score = risk_scores.get(risk_level, 0.6)

        # Adjust by VaR
        adjusted_score = min(1.0, base_score + var_1d * 10)
        return adjusted_score

    def _identify_critical_risks(self, result: Dict[str, Any]) -> List[str]:
        """Identify critical risk factors"""
        critical_risks = []

        if result.get("var_1d", 0) > 0.05:
            critical_risks.append("High volatility")
        if result.get("max_drawdown", 0) > 0.2:
            critical_risks.append("Large potential drawdown")
        if result.get("risk_level") == "high":
            critical_risks.append("High risk classification")

        return critical_risks

    def _assess_signal_strength(self, result: Dict[str, Any]) -> str:
        """Assess trading signal strength"""
        recommendation = result.get("recommendation", "HOLD")
        confidence = result.get("confidence_score", 0.5)

        if recommendation in ["BUY", "SELL"] and confidence > 0.8:
            return "strong"
        elif recommendation in ["BUY", "SELL"] and confidence > 0.6:
            return "moderate"
        else:
            return "weak"

    def _calculate_opportunity_score(self, result: Dict[str, Any]) -> float:
        """Calculate investment opportunity score"""
        recommendation = result.get("recommendation", "HOLD")
        confidence = result.get("confidence_score", 0.5)

        if recommendation == "BUY":
            return confidence
        elif recommendation == "SELL":
            return 1.0 - confidence
        else:
            return 0.5


def main():
    """Main entry point for Deep Research Agent"""

    # Initialize agent
    agent = DeepResearchAgent()

    console.print(
        Panel.fit(
            "🔬 [bold blue]Deep Research SGR Agent[/bold blue]\n"
            "Schema-Guided Reasoning for Financial Analysis\n"
            "With Adaptive NextStep Planning",
            title="Financial Deep Research",
            border_style="blue",
        )
    )

    # Example research requests
    example_requests = [
        "Deep analysis of TSLA with competitive positioning and macro factors",
        "Comprehensive research on AI sector opportunities (NVDA, AMD, GOOGL)",
        "Risk assessment and opportunity analysis for AAPL ahead of earnings",
        "Sector rotation analysis: Tech vs Healthcare investment thesis",
        "Custom research question",
    ]

    console.print("\n[bold]Example Research Questions:[/bold]")
    for i, req in enumerate(example_requests, 1):
        console.print(f"  {i}. {req}")

    while True:
        console.print("\n" + "=" * 80)
        choice = console.input("\n🔬 Select research option (1-5) or 'q' to quit: ")

        if choice.lower() == "q":
            console.print("👋 [yellow]Goodbye![/yellow]")
            break

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(example_requests):
                if choice_idx == 4:  # Custom question
                    research_question = console.input(
                        "📝 Enter your research question: "
                    )
                    symbols_input = console.input(
                        "📊 Enter symbols (comma-separated): "
                    )
                    symbols = [s.strip().upper() for s in symbols_input.split(",")]
                else:
                    research_question = example_requests[choice_idx]
                    # Extract symbols from predefined requests
                    if "TSLA" in research_question:
                        symbols = ["TSLA"]
                    elif "NVDA" in research_question:
                        symbols = ["NVDA", "AMD", "GOOGL"]
                    elif "AAPL" in research_question:
                        symbols = ["AAPL"]
                    else:
                        symbols = ["SPY", "QQQ"]

                # Create research request
                request = DeepResearchRequest(
                    research_question=research_question,
                    symbols=symbols,
                    research_scope="comprehensive",
                    max_depth=12,
                )

                # Run deep research
                agent.run_deep_research_session(request)

            else:
                console.print("❌ [red]Invalid choice. Please try again.[/red]")

        except ValueError:
            console.print("❌ [red]Invalid input. Please enter a number or 'q'.[/red]")
        except KeyboardInterrupt:
            console.print("\n⚡ [yellow]Research interrupted by user.[/yellow]")
            break
        except Exception as e:
            console.print(f"❌ [red]Error: {str(e)}[/red]")


if __name__ == "__main__":
    main()
