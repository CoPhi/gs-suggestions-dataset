VERSION := $(shell .venv/bin/python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
API_CONTAINER_NAME := gabrielegiannessi/gs-api
FRONTEND_CONTAINER_NAME := gabrielegiannessi/gs-frontend


.PHONY: requirements requirements-api build-api build-frontend run-api run \
        tag-api tag-frontend push-api push-frontend \
        release-api release-frontend release data


data:
	uv run python -m scripts.corpus_downloader
	uv run python -m scripts.split


run-api:
	uv run uvicorn backend.api.main:app --reload

requirements:
	uv export --format requirements-txt -o requirements.txt --no-hashes
	sed -i "s|file://$(PWD)/packages/|file:./packages/|g" requirements.txt

requirements-api:
	pipreqs . --force --ignore tests,migrations,docs
	mv requirements.txt requirements.txt.tmp
	grep -v "pkg-resources" requirements.txt.tmp > requirements.txt
	rm requirements.txt.tmp


build-api: requirements
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--no-cache \
		-t $(API_CONTAINER_NAME):$(VERSION) \
		-t $(API_CONTAINER_NAME):latest \
		-f ./Dockerfile \
		--push \
		.


build-frontend:
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--no-cache \
		-t $(FRONTEND_CONTAINER_NAME):$(VERSION) \
		-t $(FRONTEND_CONTAINER_NAME):latest \
		-f ./frontend/Dockerfile \
		--push \
		./frontend


run:
	docker compose up


tag-api:
	docker tag $(API_CONTAINER_NAME):latest $(API_CONTAINER_NAME):$(VERSION)


tag-frontend:
	docker tag $(FRONTEND_CONTAINER_NAME):latest $(FRONTEND_CONTAINER_NAME):$(VERSION)


push-api:
	@if ! docker image inspect $(API_CONTAINER_NAME):$(VERSION) > /dev/null 2>&1; then \
		echo "Image $(API_CONTAINER_NAME):$(VERSION) not found."; \
		exit 1; \
	fi
	docker push $(API_CONTAINER_NAME):$(VERSION)


push-frontend:
	@if ! docker image inspect $(FRONTEND_CONTAINER_NAME):$(VERSION) > /dev/null 2>&1; then \
		echo "Image $(FRONTEND_CONTAINER_NAME):$(VERSION) not found."; \
		exit 1; \
	fi
	docker push $(FRONTEND_CONTAINER_NAME):$(VERSION)


release-api: build-api
	@echo "API $(VERSION) pubblicata su Docker Hub"


release-frontend: build-frontend
	@echo "Frontend $(VERSION) pubblicato su Docker Hub"


release: release-api release-frontend
	@echo "Release $(VERSION) completata"