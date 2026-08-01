.PHONY: devcontainer-build face-tracker servos


devcontainer-build:
	docker compose -f .devcontainer/docker-compose.yml build hailo-apps-devcontainer


face-tracker: devcontainer-build
	docker compose -f .devcontainer/docker-compose.yml run --rm --entrypoint="env PYTHONPATH=/workspace/src python -m hailo_apps.scripts.face_tracker" hailo-apps-devcontainer


servos: devcontainer-build
	docker compose -f .devcontainer/docker-compose.yml run --rm --entrypoint="env PYTHONPATH=/workspace/src python -m hailo_apps.scripts.servos" hailo-apps-devcontainer


camera-memory-free:
	sudo fuser -k /dev/video* /dev/media* /dev/dma_heap/* 2>/dev/null || true
	sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
