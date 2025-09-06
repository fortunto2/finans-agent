# 🚀 SGR Multi-Agent Financial Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-orange.svg)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)

Система алгоритмической торговли, основанная на **Schema-Guided Reasoning (SGR)** подходе с использованием мультиагентной AI архитектуры. Комбинирует технический анализ, новостную аналитику, управление рисками и прогнозирование для принятия торговых решений.

## ✨ Основные возможности

### 🧠 Schema-Guided Reasoning (SGR)
- **Структурированное мышление**: пошаговое планирование и выполнение анализа
- **Azure OpenAI интеграция**: использует `client.beta.chat.completions.parse()` для структурированного вывода
- **Прозрачные решения**: каждый шаг анализа объясняется и логируется

### 🤖 Мультиагентная архитектура
- **Специализированные агенты**: каждый агент решает свою задачу
- **Координация**: агенты обмениваются информацией и формируют консенсус
- **Модульность**: легко добавлять новых агентов или улучшать существующих

### 📊 Финансовые инструменты
- **Рыночные данные**: Yahoo Finance API для OHLCV данных
- **Технический анализ**: RSI, скользящие средние, волатильность
- **Новостная аналитика**: Opoint API с продвинутым sentiment analysis
- **Управление рисками**: VaR, позиционный сайзинг, circuit breakers

### 🛡️ Безопасность и надежность
- **Paper Trading**: безопасное тестирование без реального капитала
- **Risk Management**: встроенные лимиты и контроль просадок
- **Compliance**: полный аудит-трейл всех решений

## 🚀 Быстрый старт

### Web UI (рекомендуется)
```bash
# 1. Установка зависимостей
uv sync

# 2. Настройка (скопируйте .env.example в .env и добавьте ключи)
cp .env.example .env

# 3. Запуск веб-интерфейса
./run_web_ui.sh
# или
uv run chainlit run trading_demo_app.py

# 4. Откройте в браузере: http://localhost:8000
```

### CLI версия
```bash
uv run python sgr_trading_agent.py
```

Подробные инструкции: [QUICK_START.md](QUICK_START.md)

### 🎨 Веб-интерфейс Chainlit

Красивый веб-интерфейс с пошаговой визуализацией анализа:
- **Интерактивный чат** для торговых запросов
- **Визуализация SGR шагов** в реальном времени
- **Красивое форматирование** результатов
- **Цветовая индикация** для рекомендаций

Документация: [TRADING_WEB_UI.md](TRADING_WEB_UI.md)

## 📦 Установка

### 1. Установка зависимостей

```bash
# Клонируем репозиторий
git clone https://github.com/fortunto2/finans-agent.git
cd finans-agent

# Устанавливаем зависимости через uv
uv sync
```

### 2. Настройка окружения

```bash
# Копируем пример конфигурации
cp .env.example .env

# Редактируем .env файл с вашими API ключами
# Минимально необходимые ключи:
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_ENDPOINT  
# - AZURE_OPENAI_RESOURCE_NAME
# - OPOINT_API_KEY (опционально, есть fallback на mock данные)
```

### 3. Запуск простого SGR агента

```bash
# Запускаем основной SGR агент
uv run python simple_trading_agent.py
```

### 4. Примеры задач

Система понимает естественные запросы на русском языке:

1. `Проанализируй AAPL для торговли на следующую неделю`
2. `Создай прогноз для TSLA с оценкой риска`
3. `Рекомендуй портфель из AAPL, GOOGL, MSFT с бюджетом $50,000`
4. `Оцени риски инвестиций в технологический сектор`

## 📋 Структура проекта

```
finans-agent/
├── simple_trading_agent.py    # 🎯 Основной SGR агент
├── models.py                  # 📝 Pydantic модели данных
├── settings.py               # ⚙️ Конфигурация через .env
├── api.py                    # 📰 Opoint API для новостей
├── market_data_tools.py      # 📊 Инструменты рыночной аналитики
├── sgr_trading_agent.py      # 🔧 Расширенная версия агента
├── AGENTS.md                 # 📚 Подробная документация
├── pyproject.toml           # 📦 Управление зависимостями
└── .env.example             # 🔑 Пример конфигурации
```

## 🧠 Как работает SGR

### SGR Workflow:
1. **Анализ состояния** → текущая рыночная ситуация
2. **Планирование шагов** → что нужно сделать для анализа
3. **Выполнение инструментов** → сбор данных, анализ, прогнозы
4. **Итеративное уточнение** → корректировка на основе новых данных
5. **Финальное решение** → структурированная рекомендация

### Доступные инструменты:
- `get_market_data` - получение рыночных данных
- `generate_forecast` - создание вероятностных прогнозов
- `analyze_trading` - анализ торговых возможностей
- `assess_risk` - оценка портфельных рисков
- `analyze_news` - анализ новостного фона
- `complete_analysis` - финальные рекомендации

## 📈 Пример работы

```python
# Пример SGR анализа AAPL:

🚀 Simple SGR Trading Agent
Task: Проанализируй AAPL для торговли на следующую неделю

📊 Analysis Step 1: get_market_data
📈 Market Data: AAPL - sideways trend

📊 Analysis Step 2: generate_forecast  
🔮 Forecast: 70% probability of growth

📊 Analysis Step 3: analyze_trading
🎯 Recommendation: HOLD (60% confidence)

📊 Analysis Step 4: analyze_news
📰 News Sentiment: POSITIVE (5 articles)

📊 Analysis Step 5: complete_analysis
✅ Final Decision: HOLD with 65% confidence
```

## 🔧 Расширенные возможности

### Улучшенный анализ новостей
Система реализует продвинутый анализ новостей по методологии из финансовых исследований:

- **Sentiment Analysis**: улучшенные финансовые словари
- **Causality Analysis**: определение рыночного воздействия событий  
- **Source Credibility**: взвешивание по авторитетности источника
- **Market Impact Scoring**: оценка потенциального влияния на цену

### Risk Management
- **Position Sizing**: автоматический расчет размера позиции
- **VaR Calculation**: оценка Value-at-Risk
- **Circuit Breakers**: автоматическая остановка при превышении лимитов
- **Drawdown Monitoring**: контроль максимальной просадки

### Paper Trading
- **Безопасное тестирование**: все операции в режиме симуляции
- **Реалистичные комиссии**: учет торговых издержек
- **Performance Tracking**: отслеживание результатов стратегии

## 📊 Метрики и валидация

Система отслеживает ключевые метрики эффективности:

- **Sharpe Ratio**: risk-adjusted доходность (цель >2.0)
- **Maximum Drawdown**: максимальная просадка (<10%)
- **Win Rate**: доля успешных сделок
- **Brier Score**: точность вероятностных прогнозов (<0.10)

## 🔑 Требуемые API ключи

### Обязательные:
- **Azure OpenAI**: для SGR рассуждений и структурированного вывода

### Опциональные:
- **Opoint API**: для реального анализа новостей (есть fallback на mock данные)
- **Alpha Vantage**: дополнительные рыночные данные
- **News API**: альтернативный источник новостей

## 🚨 Дисклеймер

⚠️ **Данный проект предназначен исключительно для образовательных и исследовательских целей.**

- Система работает в **Paper Trading** режиме по умолчанию
- Не является финансовым советом
- Торговля сопряжена с рисками
- Всегда консультируйтесь с финансовыми консультантами

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE) файл для деталей.

## 🤝 Участие в разработке

Мы приветствуем участие в развитии проекта! Пожалуйста:

1. Форкните репозиторий
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Коммитьте изменения (`git commit -m 'Add amazing feature'`)
4. Пушьте в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 🔗 Полезные ссылки

- [AGENTS.md](AGENTS.md) - Детальная документация системы
- [Schema-Guided Reasoning](https://abdullin.com/schema-guided-reasoning/demo) - Подход SGR
- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [Opoint API](https://opoint.com) - Источник новостных данных

---

**⭐ Если проект был полезен, поставьте звезду!**

Made with ❤️ using Azure OpenAI, Python, and lots of ☕
