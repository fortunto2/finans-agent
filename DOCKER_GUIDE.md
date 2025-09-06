# Docker Compose Guide для SGR Trading Agent

## Быстрый старт

### 1. Подготовка окружения

```bash
# Скопируйте пример переменных окружения
cp env.example .env

# Отредактируйте .env файл с вашими API ключами
nano .env
```

### 2. Запуск приложения

```bash
# Основной веб-интерфейс (без сбора данных)
docker-compose up -d

# С включенным сборщиком рыночных данных
docker-compose --profile data-collector up -d
```

### 3. Доступ к приложению

- **Локально**: http://localhost:8000
- **Через Cloudflare Tunnel**: По ссылке из вашего Cloudflare Dashboard (если настроен)

## Сервисы

### sgr-trading-agent
- **Назначение**: Основной веб-интерфейс Chainlit для торгового агента
- **Порт**: 8000
- **Volumes**: 
  - `/app/data` - кэш и данные агента
  - `/app/market_data` - рыночные данные
- **Зависимости**: Все API ключи в .env

### cloudflared (опционально)
- **Назначение**: Безопасный публичный доступ через Cloudflare Tunnel
- **Требования**: `CLOUDFLARE_TUNNEL_TOKEN` в .env
- **Зависимости**: sgr-trading-agent должен быть здоров

### market-data-collector (опционально)
- **Назначение**: Фоновый сборщик рыночных данных
- **Профиль**: `data-collector`
- **Требования**: API ключи для финансовых данных

## Команды управления

```bash
# Посмотреть логи
docker-compose logs -f sgr-trading-agent

# Перезапустить сервис
docker-compose restart sgr-trading-agent

# Остановить все сервисы
docker-compose down

# Остановить с удалением volumes
docker-compose down -v

# Пересобрать образы
docker-compose build --no-cache

# Запуск только основного сервиса
docker-compose up sgr-trading-agent

# Проверка статуса
docker-compose ps
```

## Настройка Cloudflare Tunnel

1. Создайте tunnel в Cloudflare Dashboard
2. Получите токен tunnel
3. Добавьте в .env: `CLOUDFLARE_TUNNEL_TOKEN=your_token`
4. Настройте маршрутизацию в Cloudflare Dashboard:
   - Тип: HTTP
   - URL: http://sgr-trading-agent:8000

## Мониторинг

### Проверка статуса
Проверить доступность приложения:
```bash
# Ручная проверка
curl http://localhost:8000
```

### Логи
```bash
# Все сервисы
docker-compose logs -f

# Только торговый агент
docker-compose logs -f sgr-trading-agent

# Только cloudflared
docker-compose logs -f cloudflared
```

## Переменные окружения

### Обязательные (Azure OpenAI)
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT_NAME`

### Опциональные (Финансовые API)
- `ALPHA_VANTAGE_API_KEY`
- `YAHOO_FINANCE_API_KEY`
- `FRED_API_KEY`
- `NEWS_API_KEY`

### Опциональные (Cloudflare)
- `CLOUDFLARE_TUNNEL_TOKEN`

## Troubleshooting

### Приложение не запускается
```bash
# Проверьте логи
docker-compose logs sgr-trading-agent

# Проверьте переменные окружения
docker-compose exec sgr-trading-agent env | grep AZURE
```

### Ошибки API
```bash
# Проверьте доступность API
docker-compose exec sgr-trading-agent python -c "
from settings import settings
print(f'Azure Endpoint: {settings.azure_openai_endpoint}')
print(f'Deployment: {settings.azure_openai_deployment_name}')
"
```

### Проблемы с сетью
```bash
# Проверьте сеть Docker
docker network ls
docker network inspect finans-agent_trading_network
```

## Примеры использования

После запуска откройте веб-интерфейс и попробуйте:

1. **Анализ рынка**: "Проанализируй текущую рыночную ситуацию по S&P 500"
2. **Прогноз**: "Создай прогноз на следующую неделю для AAPL"
3. **Веб-анализ**: "Проанализируй последний отчет Apple с сайта investor.apple.com"
4. **Исследование**: "Исследуй общественное мнение об ИИ в финансах"

## Безопасность

- Все API ключи хранятся в .env файле (исключен из git)
- Cloudflare Tunnel обеспечивает безопасный доступ без открытых портов
- Режим paper trading включен по умолчанию
- Встроенные лимиты риска и circuit breakers
