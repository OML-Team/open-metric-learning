JUPYTER_CMD=export TEST_RUN=1; jupyter nbconvert --to html --output-dir /tmp

DATA_DIR ?= data
RUNTIME ?= cpu
IMAGE_NAME = oml:$(RUNTIME)

# ====================================== TESTS ======================================

.PHONY: run_pytest
run_pytest:
	pytest tests --disable-warnings -sv

.PHONY: run_tests_scripts
run_tests_scripts:
	export PYTHONWARNINGS=ignore; cd tests/test_examples; rm -rf logs; python train_mock.py; rm -rf logs;
	export PYTHONWARNINGS=ignore; cd tests/test_examples; rm -rf logs; python val_mock.py; rm -rf logs;

# todo
.PHONY: test_notebooks
test_notebooks:
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
	docker run -t $(IMAGE_NAME) bash -c "make run_tests_scripts; make run_pytest"

.PHONY: upload_to_pip
upload_to_pip:
	python3 -m pip install --upgrade twine
	python3 -m pip install --upgrade build
	rm -rf dist/*
	python3 -m build
	python3 -m twine upload dist/*
