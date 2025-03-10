include .env
.PHONY: core-build devcontainer-build


core-build:
	[ -e .secrets/.env ] || touch .secrets/.env
	docker compose build hailo-apps-core

core-run:
	docker compose run hailo-apps-core


devcontainer-build: core-build
	docker compose -f .devcontainer/docker-compose.yml build hailo-apps-devcontainer
