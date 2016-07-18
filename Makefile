.PHONY: clean lint docker test

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 --exclude=lib/,bin/ .

docker:
	echo "Build docker image with name ker"
	docker build -t ker -f ./docker/Dockerfile ./docker
	echo "Now you can use it"

test:
	echo "Test on Keras + python3"
	NV_GPU=0 ./ker.sh python3 /src/test.py
