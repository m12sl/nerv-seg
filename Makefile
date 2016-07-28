.PHONY: clean lint docker test prepare stage0gi

prepare:
	python src/prepare.py --data-path data/

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 --exclude=lib/,bin/ .

docker:
	docker build -t ker -f ./docker/Dockerfile ./docker

test:
	NV_GPU=0 ./ker.sh python3 /src/test_gpu.py

stage0:
	stage0.sh

submission:
	python src/submission.py --data-path data/
