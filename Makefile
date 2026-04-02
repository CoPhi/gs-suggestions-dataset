VERSION := $(shell poetry version -s)
API_CONTAINER_NAME := gabrielegiannessi/gs-api
FRONTEND_CONTAINER_NAME := gabrielegiannessi/gs-frontend

.PHONY: requirements build-images run-api run tag-api tag-frontend push-api push-frontend

run-api:
	poetry run uvicorn backend.api.main:app --reload 

requirements:
	poetry export -f requirements.txt -o requirements.txt --without-hashes

requirements-api:
	pipreqs . --force --ignore tests,migrations,docs
	mv requirements.txt requirements.txt.tmp
	grep -v "pkg-resources" requirements.txt.tmp > requirements.txt
	rm requirements.txt.tmp

build-api: 
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