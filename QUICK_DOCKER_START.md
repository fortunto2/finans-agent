# 🚀 Быстрый запуск SGR Trading Agent в Docker

## 1-минутный старт

```bash
# 1. Подготовка
make setup
# Отредактируйте .env файл с вашими API ключами

# 2. Разработка (с auto-reload)
make dev

# 3. Продакшн (фоновый режим)
make prod
```

Приложение будет доступно на http://localhost:8000

## Команды управления

```bash
make help          # Показать все команды
make logs          # Посмотреть логи
make health        # Проверить статус
make restart       # Перезапустить
make clean         # Очистить все
```

## Варианты запуска

- `make dev` - Разработка с auto-reload
- `make prod` - Продакшн без tunnel
- `make tunnel` - С Cloudflare tunnel (нужен токен)
- `make data` - С фоновым сборщиком данных

## Минимальные требования

В .env файле обязательно укажите:
```env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your_model
```

Остальные API ключи опциональны.
