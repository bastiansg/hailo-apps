include .env
.PHONY: core-build api-build devcontainer-build


core-build:
	[ -e .secrets/.env ] || touch .secrets/.env
	docker compose build hailo-apps-core

core-run:
	docker compose run hailo-apps-core


api-build: core-build
	docker compose build hailo-apps-api

api-run: api-build
	docker compose run --rm hailo-apps-api

api-up: api-build
	docker compose up hailo-apps-api -d


devcontainer-build: core-build
	docker compose -f .devcontainer/docker-compose.yml build hailo-apps-devcontainer
