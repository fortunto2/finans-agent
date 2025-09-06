#!/usr/bin/env python3
"""
Demo script for SGR Trading Agent with Memory System
Demonstrates how the agent can use memory tools to create and maintain trading rules.
"""

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import SGR components (without Azure OpenAI dependency for demo)
from sgr_trading_agent import (
    DB,
    create_trading_rule,
    get_trading_memory,
    update_trading_rule,
    delete_trading_rule,
)

console = Console()


def demo_sgr_memory_system():
    """Demonstrate SGR memory system functionality"""

    console.print(
        Panel(
            "[bold blue]🧠 SGR Trading Agent - Memory System Demo[/bold blue]\n\n"
            "Demonstrating Schema-Guided Reasoning memory capabilities\n"
            "Based on: https://abdullin.com/schema-guided-reasoning/demo",
            title="SGR Demo",
            border_style="blue",
        )
    )

    # Initial memory state
    console.print("\n[bold yellow]📊 Initial Memory State[/bold yellow]")
    initial_memory = get_trading_memory("all")
    console.print(f"Rules: {initial_memory['data']['summary']['total_rules']}")
    console.print(
        f"Market data points: {initial_memory['data']['summary']['market_data_points']}"
    )

    # Demo 1: Creating trading rules (like SGR demo customer rules)
    console.print("\n[bold green]Demo 1: Creating Trading Rules[/bold green]")
    console.print("-" * 40)

    # Risk management rule
    rule1 = create_trading_rule(
        "Never risk more than 2% of portfolio on a single trade",
        "risk_management",
        {
            "max_risk_per_trade": 0.02,
            "stop_loss_required": True,
            "position_sizing_method": "fixed_fractional",
        },
        priority=10,
    )
    console.print(f"✅ Created risk rule: {rule1['rule_id']}")

    # Trading preference rule
    rule2 = create_trading_rule(
        "Focus on technology stocks with market cap > $10B during bull markets",
        "trading_preference",
        {
            "sectors": ["technology", "software", "semiconductors"],
            "min_market_cap": 10_000_000_000,
            "market_condition": "bullish",
            "exclude_penny_stocks": True,
        },
        priority=7,
    )
    console.print(f"✅ Created preference rule: {rule2['rule_id']}")

    # Analysis guideline
    rule3 = create_trading_rule(
        "Use multiple timeframes for technical analysis: 1D, 4H, 1H",
        "analysis_guideline",
        {
            "timeframes": ["1d", "4h", "1h"],
            "indicators": ["RSI", "MACD", "SMA_20", "SMA_50"],
            "confirmation_required": True,
        },
        priority=8,
    )
    console.print(f"✅ Created analysis rule: {rule3['rule_id']}")

    # Demo 2: Retrieving memory (like SGR demo GetCustomerData)
    console.print("\n[bold green]Demo 2: Memory Retrieval[/bold green]")
    console.print("-" * 40)

    # Get all rules
    all_rules = get_trading_memory("rules")
    console.print(f"Total active rules: {len(all_rules['data']['rules'])}")

    # Create a table to display rules
    rules_table = Table(title="Active Trading Rules")
    rules_table.add_column("Rule ID", style="cyan")
    rules_table.add_column("Type", style="green")
    rules_table.add_column("Description", style="white")
    rules_table.add_column("Priority", style="yellow")

    for rule in all_rules["data"]["rules"]:
        if rule["active"]:
            rules_table.add_row(
                rule["rule_id"][:12] + "...",
                rule["rule_type"],
                rule["description"][:50] + "..."
                if len(rule["description"]) > 50
                else rule["description"],
                str(rule["priority"]),
            )

    console.print(rules_table)

    # Filter by type (like SGR demo filtering)
    risk_rules = get_trading_memory("rules", {"rule_type": "risk_management"})
    console.print(f"\n🛡️ Risk management rules: {len(risk_rules['data']['rules'])}")

    preference_rules = get_trading_memory("rules", {"rule_type": "trading_preference"})
    console.print(
        f"📈 Trading preference rules: {len(preference_rules['data']['rules'])}"
    )

    # Demo 3: Updating rules (like SGR demo updating customer data)
    console.print("\n[bold green]Demo 3: Rule Updates[/bold green]")
    console.print("-" * 40)

    # Update risk rule with stricter parameters
    updated_rule = update_trading_rule(
        rule1["rule_id"],
        new_description="Never risk more than 1.5% of portfolio on a single trade (updated for conservative approach)",
        new_parameters={
            "max_risk_per_trade": 0.015,  # More conservative
            "stop_loss_required": True,
            "position_sizing_method": "fixed_fractional",
            "max_correlation": 0.7,  # Add correlation limit
        },
        new_priority=10,
    )
    console.print(f"✅ Updated rule: {updated_rule['rule_id']} - now more conservative")

    # Demo 4: Rule deletion (like SGR demo VoidInvoice)
    console.print("\n[bold green]Demo 4: Rule Management[/bold green]")
    console.print("-" * 40)

    # Delete analysis rule and replace with better one
    deleted_rule = delete_trading_rule(
        rule3["rule_id"],
        "Replaced with more comprehensive multi-timeframe analysis strategy",
    )
    console.print(f"🗑️ Deleted rule: {deleted_rule['rule_id']}")

    # Create improved rule
    rule4 = create_trading_rule(
        "Comprehensive multi-timeframe analysis with volume confirmation",
        "analysis_guideline",
        {
            "timeframes": ["1d", "4h", "1h", "15m"],
            "indicators": ["RSI", "MACD", "SMA_20", "SMA_50", "SMA_200", "Volume"],
            "confirmation_required": True,
            "volume_threshold": 1.5,  # 50% above average
            "divergence_check": True,
        },
        priority=9,
    )
    console.print(f"✅ Created improved rule: {rule4['rule_id']}")

    # Demo 5: Final memory state
    console.print("\n[bold green]Demo 5: Final Memory State[/bold green]")
    console.print("-" * 40)

    final_memory = get_trading_memory("all")

    # Create summary table
    summary_table = Table(title="Memory Summary")
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Count", style="green")

    summary = final_memory["data"]["summary"]
    summary_table.add_row("Total Rules", str(summary["total_rules"]))
    summary_table.add_row("Active Rules", str(summary["active_rules"]))
    summary_table.add_row("Market Data Points", str(summary["market_data_points"]))
    summary_table.add_row("Forecasts", str(summary["total_forecasts"]))
    summary_table.add_row("Trades", str(summary["total_trades"]))

    console.print(summary_table)

    # Show database structure (SGR style)
    console.print("\n[bold cyan]🏗️ SGR Database Structure (like demo DB):[/bold cyan]")
    db_structure = {
        "rules": len(DB.data["rules"]),
        "memory": {
            "market_data": len(DB.data["memory"].market_data),
            "forecasts": len(DB.data["memory"].forecasts),
            "positions": len(DB.data["memory"].positions),
            "recent_trades": len(DB.data["memory"].recent_trades),
        },
        "sessions": len(DB.data["sessions"]),
        "preferences": len(DB.data["preferences"]),
    }

    console.print(json.dumps(db_structure, indent=2))

    # Memory usage patterns
    console.print("\n[bold yellow]💡 Memory Usage Patterns (SGR style):[/bold yellow]")
    console.print("• Rules are created and stored persistently in memory")
    console.print(
        "• Memory can be filtered by type (risk_management, trading_preference, etc.)"
    )
    console.print("• Rules can be updated with new parameters and priorities")
    console.print("• Rules can be deleted with reason logging for audit trail")
    console.print("• All operations return structured responses for agent processing")
    console.print(
        "• Compatible with SGR pattern from https://abdullin.com/schema-guided-reasoning/demo"
    )

    console.print(
        Panel(
            "[bold green]✅ SGR Memory System Demo Completed![/bold green]\n\n"
            "The trading agent now has persistent memory capabilities:\n"
            "• Rule creation and management\n"
            "• Structured memory retrieval\n"
            "• Audit trail for all changes\n"
            "• SGR-compatible database structure\n\n"
            "Ready for use with Azure OpenAI structured output!",
            title="Demo Complete",
            border_style="green",
        )
    )

    return final_memory


if __name__ == "__main__":
    try:
        final_state = demo_sgr_memory_system()
        console.print("\n🎉 [bold green]Demo completed successfully![/bold green]")

        # Show how to use in SGR agent
        console.print("\n[bold blue]💡 Usage in SGR Trading Agent:[/bold blue]")
        console.print("To use these memory tools in the SGR agent, include tasks like:")
        console.print(
            "• 'Создай правило: никогда не рискуй более 2% портфеля на одну сделку'"
        )
        console.print("• 'Покажи все мои правила управления рисками'")
        console.print("• 'Обнови правило анализа - добавь проверку объемов'")
        console.print("• 'Удали устаревшее правило и объясни почему'")

    except Exception as e:
        console.print(f"❌ [bold red]Demo failed: {e}[/bold red]")
        import traceback

        traceback.print_exc()
