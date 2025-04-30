VERSION := $(shell poetry version | cut -d ' ' -f 2)
API_CONTAINER_NAME := gabrielegiannessi/gs-api
FRONTEND_CONTAINER_NAME := gabrielegiannessi/gs-frontend

train: 
	poetry run python -m train.training

eval: 
	poetry run python -m eval.KFold

requirements:
	poetry export -f requirements.txt -o requirements.txt --without-hashes

build: 
	docker compose build --no-cache
	
run: 
	docker compose up

tag-api:
	docker tag $(API_CONTAINER_NAME):latest	 $(API_CONTAINER_NAME):$(VERSION)

tag-frontend:
	docker tag $(FRONTEND_CONTAINER_NAME):latest $(FRONTEND_CONTAINER_NAME):$(VERSION)

push-api:
	docker push $(API_CONTAINER_NAME):$(VERSION)

push-frontend:
	docker push $(FRONTEND_CONTAINER_NAME):$(VERSION)

.PHONY:
	requirements build train eval run tag-api tag-frontend push-api push-frontend