.PHONY: build run

build:
	docker build . -f Dockerfile -t deffe:dev

run:
	docker run -it --rm \
		--mount source=deffe-workspace,target=/workspace \
		deffe:dev
