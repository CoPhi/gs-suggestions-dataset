VERSION := $(shell poetry version -s)
API_CONTAINER_NAME := gabrielegiannessi/gs-api
FRONTEND_CONTAINER_NAME := gabrielegiannessi/gs-frontend

.PHONY: requirements build-images run-api training assessment run tag-api tag-frontend push-api push-frontend

training: 
	poetry run python -m train.training

assessment: 
	poetry run python -m eval.KFold

run-api:
	poetry run uvicorn api.main:app --reload 

requirements:
	poetry export -f requirements.txt -o requirements.txt --without-hashes

build-images: 
	docker compose build --no-cache
	
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