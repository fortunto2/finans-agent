# SGR Trading Agent - Docker Management

.PHONY: help build up down logs restart clean dev prod tunnel health

# Default help
help:
	@echo "SGR Trading Agent - Docker Commands"
	@echo ""
	@echo "Development:"
	@echo "  dev      - Run development version (auto-reload)"
	@echo "  dev-logs - Show development logs"
	@echo ""
	@echo "Production:"
	@echo "  prod     - Run production version"
	@echo "  tunnel   - Run with Cloudflare tunnel"
	@echo "  data     - Run with market data collector"
	@echo ""
	@echo "Management:"
	@echo "  build    - Build Docker images"
	@echo "  up       - Start services"
	@echo "  down     - Stop services"
	@echo "  restart  - Restart services"
	@echo "  logs     - Show logs"
	@echo "  health   - Check service health"
	@echo "  clean    - Clean up containers and volumes"
	@echo ""
	@echo "Setup:"
	@echo "  setup    - Initial setup (copy env file)"

# Setup
setup:
	@if [ ! -f .env ]; then \
		cp env.example .env; \
		echo "✅ Created .env file from env.example"; \
		echo "⚠️  Please edit .env with your API keys"; \
	else \
		echo "✅ .env file already exists"; \
	fi

# Development
dev: setup
	docker-compose -f docker-compose.dev.yml up --build

dev-logs:
	docker-compose -f docker-compose.dev.yml logs -f

dev-down:
	docker-compose -f docker-compose.dev.yml down

# Production
prod: setup
	docker-compose up -d

tunnel: setup
	@if grep -q "CLOUDFLARE_TUNNEL_TOKEN=your_cloudflare_tunnel_token" .env; then \
		echo "❌ Please set CLOUDFLARE_TUNNEL_TOKEN in .env file"; \
		exit 1; \
	fi
	docker-compose up -d

data: setup
	docker-compose --profile data-collector up -d

# Build
build:
	docker-compose build --no-cache

# Management
up: setup
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

health:
	@echo "Checking service status..."
	@curl -f http://localhost:8000 2>/dev/null && echo "✅ Trading Agent: Responding" || echo "❌ Trading Agent: Not responding"
	@docker-compose ps

# Cleanup
clean:
	docker-compose down -v
	docker system prune -f
	docker volume prune -f

# Quick commands
start: up
stop: down
rebuild: clean build up
