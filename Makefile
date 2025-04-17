VERSION := $(shell poetry version | cut -d ' ' -f 2)
CONTAINER := gabrielegiannessi/gs-api

requirements:
	poetry export -f requirements.txt -o requirements.txt --without-hashes

container: requirements
	docker build -t $(CONTAINER):$(VERSION) -t $(CONTAINER):latest .

.PHONY: requirements container
