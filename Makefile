VERSION := $(shell poetry version | cut -d ' ' -f 2)
CONTAINER_NAME := gabrielegiannessi/gs-api


train: 
	poetry run python -m train.training

eval: 
	poetry run python -m eval.KFold
	
requirements:
	poetry export -f requirements.txt -o requirements.txt --without-hashes

container: requirements
	docker build -t $(CONTAINER_NAME):$(VERSION) -t $(CONTAINER_NAME):latest .

.PHONY: requirements container 

run: 
	docker run -p 8000:8000 $(CONTAINER_NAME):$(VERSION)
