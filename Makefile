JUPYTER_CMD=export TEST_RUN=1; jupyter nbconvert --to html --output-dir /tmp


.PHONY: run_mock_scripts
run_mock_scripts:
	export PYTHONWARNINGS=ignore; cd tests/test_examples; rm -rf logs; python train_mock.py; rm -rf logs;
	export PYTHONWARNINGS=ignore; cd tests/test_examples; rm -rf logs; python val_mock.py; rm -rf logs;

.PHONY: run_tests
run_tests:
	pytest tests --disable-warnings -sv

.PHONY: run_precommit
run_precommit:
	pre-commit install && pre-commit run -a

.PHONY: test_notebooks
test_notebooks:
	$(JUPYTER_CMD) --execute examples/visualization.ipynb

.PHONY: check_converters
check_converters:
	clear
	export PYTHONWARNINGS=ignore; python examples/cub/convert_cub.py        --dataset_root  /nydl/data/CUB_200_2011
	export PYTHONWARNINGS=ignore; python examples/sop/convert_sop.py        --dataset_root  /nydl/data/Stanford_Online_Products
	export PYTHONWARNINGS=ignore; python examples/cars/convert_cars.py      --dataset_root  /nydl/data/CARS196
	export PYTHONWARNINGS=ignore; python examples/inshop/convert_inshop.py  --dataset_root  /nydl/data/DeepFashion_InShop


RUNTIME ?= cpu
IMAGE_NAME = oml:$(RUNTIME)

.PHONY: docker_build
docker_build:
	DOCKER_BUILDKIT=1 docker build --build-arg RUNTIME=$(RUNTIME) -t $(IMAGE_NAME) -f ci/Dockerfile .


.PHONY: docker_tests
docker_tests:
	docker run -t $(IMAGE_NAME) bash -c "make run_mock_scripts; make run_tests"
