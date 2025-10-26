.PHONY: up up-detach down logs ps build clean help

# docker compose wrapper targets to manage the container stack
COMPOSE ?= docker compose

help:
	@echo "Available targets:"
	@echo "  make up        Start the stack in the foreground (builds first)"
	@echo "  make up-detach Start the stack in the background (builds first)"
	@echo "  make build     Build the app image without starting containers"
	@echo "  make down      Stop the stack and remove containers"
	@echo "  make logs      Tail logs for all services or SERVICE=name"
	@echo "  make ps        Show container status"
	@echo "  make clean     Remove containers and the postgres volume"
	@echo "  make help      Show this help message"

up: build
	$(COMPOSE) up $(UP_FLAGS)

up-detach: UP_FLAGS += -d
up-detach: up

build:
	$(COMPOSE) build $(SERVICE)

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs --follow $(SERVICE)

ps:
	$(COMPOSE) ps

clean:
	$(COMPOSE) down -v
