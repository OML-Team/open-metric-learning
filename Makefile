JUPYTER_CMD=export TEST_RUN=1; jupyter nbconvert --to html --output-dir /tmp

DATA_DIR ?= data
RUNTIME ?= cpu
IMAGE_NAME = oml:$(RUNTIME)

# ====================================== TESTS ======================================

.PHONY: download_mock_dataset
download_mock_dataset:
	python oml/utils/download_mock_dataset.py

.PHONY: run_tests
run_tests: download_mock_dataset
	pytest --disable-warnings -sv tests
	$(JUPYTER_CMD) --execute examples/visualization.ipynb

.PHONY: test_converters
test_converters:
	clear
	export PYTHONWARNINGS=ignore; python examples/cub/convert_cub.py        --dataset_root  {DATA_DIR}/CUB_200_2011
	export PYTHONWARNINGS=ignore; python examples/sop/convert_sop.py        --dataset_root  {DATA_DIR}/Stanford_Online_Products
	export PYTHONWARNINGS=ignore; python examples/cars/convert_cars.py      --dataset_root  {DATA_DIR}/CARS196
	export PYTHONWARNINGS=ignore; python examples/inshop/convert_inshop.py  --dataset_root  {DATA_DIR}/DeepFashion_InShop

.PHONY: run_precommit
run_precommit:
	pre-commit install && pre-commit run -a

# ====================================== INFRASTRUCTURE =============================

.PHONY: docker_build
docker_build:
	DOCKER_BUILDKIT=1 docker build --build-arg RUNTIME=$(RUNTIME) -t $(IMAGE_NAME) -f ci/Dockerfile .

.PHONY: docker_tests
docker_tests:
	docker run -t $(IMAGE_NAME) make run_tests

.PHONY: upload_to_pip
upload_to_pip:
	python -m pip install --upgrade pip
	python3 -m pip install --upgrade twine
	pip install --upgrade pip setuptools wheel
	rm -rf dist build open_metric_learning.egg-info
	python3 setup.py sdist bdist_wheel
	twine upload dist/*
