.PHONY: clean lint docker test prepare

prepare:
	python src/data.py --data_path data/

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 --exclude=lib/,bin/ .

docker:
	docker build -t ker -f ./docker/Dockerfile ./docker

test:
	NV_GPU=0 ./ker.sh python /src/test_gpu.py
