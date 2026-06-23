.PHONY: devcontainer-build


devcontainer-build:
	docker compose -f .devcontainer/docker-compose.yml build hailo-apps-devcontainer


camera-memory-free:
	sudo fuser -k /dev/video* /dev/media* /dev/dma_heap/* 2>/dev/null || true
	sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
