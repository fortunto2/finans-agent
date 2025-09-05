# AGENTS.md

## Project Overview

This is a Schema-Guided Reasoning (SGR) multi-agent system for financial forecasting and algorithmic trading, built following the SGR pattern from [abdullin.com/schema-guided-reasoning](https://abdullin.com/schema-guided-reasoning/demo). The system uses Azure OpenAI with structured output to coordinate specialized financial agents for market prediction, risk management, and automated trading decisions.

## Setup Commands

- Install dependencies: `uv sync`
- Run the SGR trading agent: `uv run python sgr_trading_agent.py`
- Run market data collector: `uv run python market_data_collector.py`
- Run backtesting suite: `uv run python backtester.py`

## Environment Configuration

Create a `.env` file in the project root with required credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=model-router
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_REGION=eastus2
AZURE_OPENAI_RESOURCE_NAME=your-resource-name

# Financial Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
YAHOO_FINANCE_API_KEY=your_yahoo_finance_key
FRED_API_KEY=your_fred_api_key
NEWS_API_KEY=your_news_api_key

# Trading Platforms
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
INTERACTIVE_BROKERS_API_KEY=your_ib_key

# Prophet Arena (Forecasting Platform)
PROPHET_ARENA_API_KEY=your_prophet_arena_key

# Risk Management
MAX_POSITION_SIZE=0.05  # 5% max position size
MAX_DAILY_DRAWDOWN=0.02  # 2% max daily drawdown
ENABLE_PAPER_TRADING=true
```

## Code Structure

- `sgr_trading_agent.py` - Main SGR trading agent with Azure OpenAI structured output
- `models.py` - Pydantic models for financial data (MarketData, Trade, ForecastResult, etc.)
- `settings.py` - Pydantic Settings for configuration management
- `agents/` - Individual specialized trading agents
  - `data_intelligence_agent.py` - Data collection and processing
  - `forecasting_agents.py` - Bull/Bear/Technical/Sentiment agents
  - `risk_management_agent.py` - Risk assessment and position sizing
  - `execution_agent.py` - Order execution and portfolio management
  - `compliance_agent.py` - Regulatory compliance and audit
- `market_data/` - Market data collection and storage
- `backtesting/` - Backtesting framework and historical analysis
- `forecasting/` - Prophet Arena integration and forecast validation

## Agent Tools Available

The SGR trading system has the following specialized agents and tools:

### Data Intelligence Agents
- `market_data_collector` - Real-time market data from multiple sources
- `news_sentiment_analyzer` - Financial news analysis and sentiment scoring
- `economic_indicator_tracker` - Macroeconomic data monitoring (FRED, etc.)
- `alternative_data_processor` - Social media, satellite, blockchain data analysis

### Forecasting Agents
- `bullish_agent` - Optimistic market outlook and long signals
- `bearish_agent` - Pessimistic market outlook and short signals  
- `technical_agent` - Technical analysis and chart pattern recognition
- `fundamental_agent` - Financial statement and valuation analysis
- `sentiment_agent` - Market sentiment and positioning analysis
- `macro_agent` - Macroeconomic and geopolitical factor analysis

### Risk Management & Execution
- `risk_manager` - Position sizing, VaR calculation, drawdown monitoring
- `portfolio_manager` - Asset allocation and position optimization
- `execution_agent` - Smart order routing and execution optimization
- `compliance_agent` - Regulatory compliance and audit trail

### Forecasting & Validation Tools
- `prophet_arena_client` - Integration with forecasting benchmark platform
- `forecast_aggregator` - Multi-agent forecast consensus building
- `calibration_monitor` - Brier score tracking and model performance
- `backtest_engine` - Historical strategy validation

### Input Validation
All tools use Pydantic schemas for input validation:
- `MarketDataRequest` - Real-time data queries
- `ForecastRequest` - Prediction parameters and timeframes
- `TradeRequest` - Order execution specifications
- `RiskAssessmentRequest` - Risk analysis parameters
- `BacktestRequest` - Historical simulation settings
- `ComplianceCheckRequest` - Regulatory validation parameters

## Development Guidelines

### Code Style

**IMPORTANT**: All code, comments, documentation, and variable names must be written in English, regardless of the communication language used in conversations.

- Use Python 3.10+ with type hints
- Follow Pydantic v2 patterns for data models
- Use Azure OpenAI structured output with `client.beta.chat.completions.parse()`
- Implement SGR pattern: current_state → plan_remaining_steps → function selection
- Use Rich console for pretty output formatting
- **All code and comments must be in English**

### Testing the Trading Agent

Example tasks to test the system:

1. "Проанализируй текущую рыночную ситуацию по S&P 500"
2. "Создай прогноз на следующую неделю для AAPL с оценкой риска"
3. "Рекомендуй портфель акций с бюджетом $100,000 и толерантностью к риску 15%"
4. "Проведи бэктест стратегии momentum на данных за последний год"
5. "Оцени влияние решения ФРС по ставкам на технологический сектор"

Example prediction tasks for Prophet Arena validation:

1. "Будет ли курс EURUSD выше 1.10 через месяц?"
2. "Превысит ли волатильность VIX уровень 25 в ближайшие 2 недели?"
3. "Вырастет ли цена нефти WTI более чем на 5% в следующем квартале?"

### Important Implementation Details

- The system uses `SGRTradingResponse` Pydantic model with structured output format
- All tool responses are JSON-formatted for consistency
- **Multi-Agent Architecture**: Specialized agents coordinate through message passing
- **Prophet Arena Integration**: Real-time forecast validation and Brier score tracking
- **Risk Management**: Built-in position sizing, VaR calculation, and circuit breakers
- **Paper Trading**: Safe testing environment before live capital deployment
- **Compliance by Design**: Full audit trail and regulatory compliance features
- **Real-time Adaptation**: Continuous learning from market feedback
- **High Availability**: Multi-cloud deployment with automatic failover

### Git Workflow

- Use `git add --no-verify` and `git commit --no-verify` for this project
- Environment variables are loaded from `.env` file (excluded from git)
- Market data and model checkpoints are stored locally and backed up regularly

## Market Data Collection

### Overview

Multi-source financial data pipeline for real-time and historical market data collection. Integrates with major financial data providers and alternative data sources.

### Features

- **Multi-source integration** - Yahoo Finance, Alpha Vantage, FRED, Bloomberg Terminal
- **Real-time streaming** - WebSocket connections for live price feeds
- **Alternative data** - Social media sentiment, news analysis, satellite data
- **Historical data** - Deep historical datasets for backtesting and training
- **Data validation** - Quality checks, anomaly detection, and data cleansing

### Usage Commands

```bash
# Real-time market data collection
uv run python market_data_collector.py --symbols AAPL,GOOGL,MSFT --real-time

# Historical data download
uv run python market_data_collector.py --symbols SPY --start-date 2020-01-01 --end-date 2024-01-01

# News and sentiment data
uv run python news_collector.py --sources reuters,bloomberg,cnbc --sentiment-analysis

# Alternative data collection
uv run python alt_data_collector.py --twitter-sentiment --reddit-sentiment --blockchain-metrics
```

### Output Structure

**Market Data**: `market_data/YYYY-MM-DD/`
- Time-series data in Parquet format for efficient querying
- Separate files for OHLCV, orderbook, and trade data
- Real-time data stored in InfluxDB for fast access

**News & Sentiment**: `news_data/YYYY-MM-DD/`
- Processed news articles with sentiment scores
- Social media data aggregated by symbol and timeframe
- Alternative data sources with relevance scoring

### Data Fields Captured

**Market Data:**
- `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`
- `bid`, `ask`, `spread`, `market_cap`, `volatility`
- Level 2 order book data for supported exchanges

**News & Sentiment:**
- `headline`, `content`, `source`, `sentiment_score`, `relevance_score`
- `entities_mentioned`, `impact_assessment`, `credibility_rating`

**Alternative Data:**
- Social media mentions, engagement metrics, trending topics
- Satellite imagery analysis for commodity tracking
- Blockchain metrics for cryptocurrency analysis

**Economic Indicators:**
- `indicator_name`, `value`, `forecast`, `previous`, `impact_level`
- Central bank decisions, employment data, inflation metrics

### Technical Details

**Data Pipeline Architecture:**
- Built with Apache Kafka for real-time streaming
- Pydantic models for data validation and serialization
- Custom adapters for each data source API
- InfluxDB for time-series storage, PostgreSQL for metadata

**Configuration:**
- `REAL_TIME_ENABLED=True/False` - Enable/disable live data streaming
- `DATA_RETENTION_DAYS=365` - Historical data retention policy
- `QUALITY_THRESHOLD=0.95` - Data quality minimum threshold
- Rate limiting and API quota management

## Prophet Arena Integration

### Forecasting Validation Platform

The system integrates with Prophet Arena (similar to ForecastBench) for objective forecast validation:

- **Real-time Benchmarking**: All predictions tested on future events, no data leakage
- **Brier Score Tracking**: Continuous calibration monitoring (target: <0.10)
- **Multi-domain Questions**: 1000+ forecast questions across economics, geopolitics, markets
- **Competitive Validation**: AI performance compared against human expert forecasters

### Data Sources

- **Primary Markets**: NYSE, NASDAQ, CME, ICE via official APIs
- **Alternative Data**: Twitter/X sentiment, Reddit discussions, Google Trends
- **Economic Indicators**: Federal Reserve (FRED), Bureau of Labor Statistics
- **News Sources**: Reuters, Bloomberg, Financial Times, MarketWatch
- **Blockchain Data**: On-chain metrics for cryptocurrency analysis

## Security & Risk Management

### Financial Security
- **API Key Management**: All trading API keys encrypted and stored securely
- **Position Limits**: Maximum position size limits (default: 5% of portfolio)
- **Circuit Breakers**: Automatic trading halt on excessive losses
- **Paper Trading Mode**: Safe testing environment before live capital

### Technical Security
- **Multi-Agent Isolation**: Each agent runs in sandboxed containers
- **Cryptographic Signatures**: Message authenticity verification between agents
- **Penetration Testing**: Regular security audits and vulnerability assessments
- **Real-time Monitoring**: 24/7 system health and anomaly detection

### Regulatory Compliance
- **Audit Trail**: Complete decision logging for regulatory review
- **SEC/CFTC Compliance**: Built-in regulatory reporting features
- **EU AI Act Ready**: Explainable AI decisions for European markets
- **Risk Disclosure**: Transparent risk metrics and performance reporting

## Troubleshooting

### Common Issues
- **Agent Communication Failures**: Check message bus connectivity and agent status
- **Data Feed Interruptions**: Verify API keys and data source availability
- **Model Calibration Drift**: Monitor Brier scores and retrain if performance degrades
- **Execution Delays**: Check broker API status and network latency

### Performance Monitoring
- **Sharpe Ratio Target**: Maintain >2.0 for institutional quality
- **Maximum Drawdown**: Alert if >10% portfolio loss
- **Forecast Accuracy**: Brier score monitoring (target: <0.10)
- **System Uptime**: 99.9% availability target with automatic failover

## Dependencies

Key dependencies managed via `pyproject.toml`:

**Core SGR Trading System:**
- `openai>=1.12.0` - Azure OpenAI client with structured output support
- `pydantic>=2.0.0` - Data models and settings validation
- `pydantic-settings>=2.0.0` - Environment-based configuration
- `rich>=13.7.0` - Console formatting and pretty output
- `python-dotenv>=1.0.0` - Environment variable loading

**Financial Data & APIs:**
- `yfinance>=0.2.0` - Yahoo Finance API client
- `alpha-vantage>=2.3.1` - Alpha Vantage API wrapper
- `fredapi>=0.5.0` - Federal Reserve Economic Data API
- `ccxt>=4.0.0` - Cryptocurrency exchange APIs
- `ib-insync>=0.9.86` - Interactive Brokers API

**Data Processing & Storage:**
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `polars>=0.20.0` - Fast DataFrame operations
- `influxdb-client>=1.38.0` - Time-series database client
- `kafka-python>=2.0.2` - Apache Kafka client

**Machine Learning & Analysis:**
- `scikit-learn>=1.3.0` - Machine learning utilities
- `scipy>=1.11.0` - Scientific computing
- `ta-lib>=0.4.28` - Technical analysis library
- `statsmodels>=0.14.0` - Statistical modeling

**Risk Management & Backtesting:**
- `quantlib>=1.32` - Quantitative finance library
- `backtrader>=1.9.78` - Backtesting framework
- `pyfolio>=0.9.2` - Portfolio performance analysis

## Project Background

This multi-agent trading system implements Schema-Guided Reasoning (SGR) pattern for financial markets:

### SGR Trading Pipeline

1. **Market State Analysis**: Multi-agent assessment of current market conditions
2. **Forecast Generation**: Specialized agents create probabilistic predictions
3. **Consensus Building**: Forecast aggregator combines and calibrates predictions
4. **Risk Assessment**: Risk management agent evaluates potential positions
5. **Execution Planning**: Smart execution agent optimizes order placement
6. **Continuous Learning**: Performance feedback updates agent models

### Key Innovations

- **Probabilistic Forecasting**: Focus on probability distributions, not point predictions
- **Multi-Agent Specialization**: Bull/bear agents, technical analysis, fundamental analysis
- **Real-time Validation**: Prophet Arena integration for objective performance measurement
- **Risk-First Design**: Built-in position sizing and circuit breakers
- **Explainable Decisions**: Every trade backed by transparent reasoning

### Market Philosophy

Following the SGR approach, the system treats financial markets as the ultimate test for artificial intelligence. Success requires:
- **Predictive Power**: Ability to forecast future events with calibrated confidence
- **Risk Management**: Sophisticated understanding of uncertainty and position sizing
- **Adaptive Learning**: Continuous improvement from market feedback
- **Regulatory Compliance**: Full transparency and auditability for institutional use

The system aims to achieve superhuman forecasting accuracy (Brier score <0.09) while maintaining institutional-grade risk management (Sharpe ratio >2.0, max drawdown <10%).
