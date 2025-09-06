#!/usr/bin/env python3
"""
Settings for Financial Trading Agent

Environment-based configuration using Pydantic Settings.
Loads configuration from .env file and environment variables.
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class TradingAgentSettings(BaseSettings):
    """Trading Agent configuration settings"""

    # Azure OpenAI Configuration
    azure_openai_api_key: str = Field(..., description="Azure OpenAI API key")
    azure_openai_endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    azure_openai_api_version: str = Field(
        default="2024-12-01-preview", description="Azure OpenAI API version"
    )
    azure_openai_deployment_name: str = Field(
        default="model-router", description="Azure OpenAI deployment name"
    )
    azure_openai_region: str = Field(
        default="eastus2", description="Azure OpenAI region"
    )
    azure_openai_resource_name: str = Field(
        ..., description="Azure OpenAI resource name"
    )

    # Financial Data APIs
    alpha_vantage_api_key: Optional[str] = Field(
        default=None, description="Alpha Vantage API key"
    )
    yahoo_finance_api_key: Optional[str] = Field(
        default=None, description="Yahoo Finance API key"
    )
    fred_api_key: Optional[str] = Field(
        default=None, description="FRED (Federal Reserve) API key"
    )
    news_api_key: Optional[str] = Field(default=None, description="News API key")
    opoint_api_key: Optional[str] = Field(default=None, description="Opoint API key")
    firecrawl_api_key: Optional[str] = Field(
        default=None, description="Firecrawl API key for web scraping"
    )

    # Trading Platforms (Demo/Paper Trading)
    binance_api_key: Optional[str] = Field(default=None, description="Binance API key")
    binance_secret_key: Optional[str] = Field(
        default=None, description="Binance secret key"
    )
    interactive_brokers_api_key: Optional[str] = Field(
        default=None, description="Interactive Brokers API key"
    )

    # Prophet Arena (Forecasting Platform)
    prophet_arena_api_key: Optional[str] = Field(
        default=None, description="Prophet Arena API key"
    )

    # Risk Management Settings
    max_position_size: float = Field(
        default=0.05, description="Maximum position size as fraction of portfolio"
    )
    max_daily_drawdown: float = Field(
        default=0.02, description="Maximum daily drawdown (2%)"
    )
    enable_paper_trading: bool = Field(
        default=True, description="Enable paper trading mode"
    )

    # Agent Configuration
    max_sgr_steps: int = Field(default=10, description="Maximum SGR reasoning steps")
    default_forecast_horizon: str = Field(
        default="1_week", description="Default forecast horizon"
    )
    risk_tolerance: str = Field(default="medium", description="Default risk tolerance")

    # Data Configuration
    real_time_enabled: bool = Field(
        default=False, description="Enable real-time data streaming"
    )
    data_retention_days: int = Field(
        default=365, description="Data retention period in days"
    )
    quality_threshold: float = Field(
        default=0.95, description="Data quality minimum threshold"
    )

    # Performance Targets
    target_sharpe_ratio: float = Field(
        default=2.0, description="Target Sharpe ratio for institutional quality"
    )
    max_drawdown_threshold: float = Field(
        default=0.10, description="Maximum acceptable drawdown (10%)"
    )
    target_brier_score: float = Field(
        default=0.09, description="Target Brier score for forecasting"
    )

    # Supported Symbols for Demo
    default_symbols: List[str] = Field(
        default=[
            "AAPL",
            "GOOGL",
            "MSFT",
            "TSLA",
            "AMZN",  # Tech stocks
            "SPY",
            "QQQ",
            "VTI",  # ETFs
            "BTC-USD",
            "ETH-USD",  # Crypto
        ],
        description="Default symbols for analysis",
    )

    # News Sources
    default_news_sources: List[str] = Field(
        default=["reuters", "bloomberg", "cnbc", "marketwatch"],
        description="Default news sources for sentiment analysis",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = TradingAgentSettings()


def get_available_data_sources() -> dict:
    """Get available data sources based on configured API keys"""
    sources = {"market_data": [], "news": [], "economic": []}

    # Check market data sources
    if settings.alpha_vantage_api_key:
        sources["market_data"].append("alpha_vantage")
    if settings.yahoo_finance_api_key:
        sources["market_data"].append("yahoo_finance")

    # Always available (free)
    sources["market_data"].append("yahoo_finance_free")

    # Check news sources
    if settings.news_api_key:
        sources["news"].append("news_api")
    if settings.opoint_api_key:
        sources["news"].append("opoint")
    if settings.firecrawl_api_key:
        sources["news"].append("firecrawl_web_scraping")

    # Check economic data
    if settings.fred_api_key:
        sources["economic"].append("fred")

    return sources


def validate_required_keys() -> tuple[bool, List[str]]:
    """Validate that required API keys are present"""
    missing_keys = []

    # Required keys
    if not settings.azure_openai_api_key:
        missing_keys.append("AZURE_OPENAI_API_KEY")
    if not settings.azure_openai_endpoint:
        missing_keys.append("AZURE_OPENAI_ENDPOINT")
    if not settings.azure_openai_resource_name:
        missing_keys.append("AZURE_OPENAI_RESOURCE_NAME")

    return len(missing_keys) == 0, missing_keys


def get_paper_trading_config() -> dict:
    """Get paper trading configuration"""
    return {
        "enabled": settings.enable_paper_trading,
        "initial_capital": 100000.0,  # $100k starting capital
        "max_position_size": settings.max_position_size,
        "max_daily_drawdown": settings.max_daily_drawdown,
        "commission_rate": 0.001,  # 0.1% commission
    }


def get_risk_management_config() -> dict:
    """Get risk management configuration"""
    return {
        "max_position_size": settings.max_position_size,
        "max_daily_drawdown": settings.max_daily_drawdown,
        "target_sharpe_ratio": settings.target_sharpe_ratio,
        "max_drawdown_threshold": settings.max_drawdown_threshold,
        "var_confidence_level": 0.95,  # 95% confidence for VaR calculations
        "stress_test_scenarios": [
            "market_crash_2008",
            "covid_crash_2020",
            "dotcom_bubble_2000",
        ],
    }


if __name__ == "__main__":
    # Quick settings validation
    print("Financial Trading Agent Settings")
    print("=" * 40)

    is_valid, missing = validate_required_keys()
    if is_valid:
        print("✓ All required API keys configured")
    else:
        print(f"✗ Missing required keys: {', '.join(missing)}")

    available_sources = get_available_data_sources()
    print(f"\nAvailable data sources:")
    for category, sources in available_sources.items():
        print(f"  {category}: {', '.join(sources) if sources else 'None'}")

    print(
        f"\nPaper trading: {'Enabled' if settings.enable_paper_trading else 'Disabled'}"
    )
    print(f"Max position size: {settings.max_position_size:.1%}")
    print(f"Target Sharpe ratio: {settings.target_sharpe_ratio:.1f}")
    print("Default symbols: " + ", ".join(settings.default_symbols[:5]) + "...")
