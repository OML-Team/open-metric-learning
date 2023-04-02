JUPYTER_CMD=export TEST_RUN=1; jupyter nbconvert --to html --output-dir /tmp

RUNTIME ?= cpu
IMAGE_NAME ?= omlteam/oml:$(RUNTIME)

README_FILE ?= README.md

OML_VERSION=$(shell cat oml/__init__.py | sed 's,.*__version__ = "\(.*\)".*,\1,')

# Note, we use the markdown files below to build the documentation (readthedocs).
# The problem with the documentation is that .rst files have to have their own headers in a special format.
# So, to avoid duplicating headers in the documentation we removed them from markdown files.
.PHONY: build_readme
build_readme:
	rm -f ${README_FILE}
	touch ${README_FILE}
	# Header
	cat docs/readme/header.md >> ${README_FILE}
	# FAQ
	echo "\n## FAQ\n" >> ${README_FILE}
	cat docs/readme/faq.md >> ${README_FILE}
	# Documentation
	echo "\n## [Documentation](https://open-metric-learning.readthedocs.io/en/latest/index.html)\n" >> ${README_FILE}
	cat docs/readme/documentation.md >> ${README_FILE}
	# Installation
	echo "\n## Installation\n" >> ${README_FILE}
	cat docs/readme/installation.md >> ${README_FILE}
	# Python examples
	echo "\n## Examples\n" >> ${README_FILE}
	cat docs/readme/examples_source/extractor/train.md >> ${README_FILE}
	cat docs/readme/examples_source/extractor/retrieval_usage.md >> ${README_FILE}
	# Pipelines
	echo "\n## Pipelines\n" >> ${README_FILE}
	cat docs/readme/get_started_config.md >> ${README_FILE}
	# Zoo
	echo "\n## Zoo\n" >>${README_FILE}
	cat docs/readme/zoo.md >> ${README_FILE}
	echo "\n### How to use models from Zoo?\n" >> ${README_FILE}
	cat docs/readme/examples_source/zoo/models_usage.md >> ${README_FILE}
	# Contributing
	echo "\n## Contributing guide\n" >> ${README_FILE}
	cat docs/readme/contributing.md >> ${README_FILE}
	# Acknowledgments
	echo "\n## Acknowledgments\n" >> ${README_FILE}
	cat docs/readme/acknowledgments.md >> ${README_FILE}

# ====================================== TESTS ======================================

.PHONY: download_mock_dataset
download_mock_dataset:
	python oml/utils/download_mock_dataset.py

.PHONY: run_tests
run_tests: download_mock_dataset
	pytest --disable-warnings -sv tests
	pytest --disable-warnings --doctest-modules --doctest-continue-on-failure -sv oml
	$(JUPYTER_CMD) --execute pipelines/visualization.ipynb

.PHONY: test_converters
test_converters:
	clear
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_cub.py     --dataset_root  data/CUB_200_2011
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_sop.py     --dataset_root  data/Stanford_Online_Products
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_cars.py    --dataset_root  data/CARS196
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_inshop.py  --dataset_root  data/DeepFashion_InShop

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

.PHONY: build_wheel
build_wheel:
	python -m pip install --upgrade pip
	python3 -m pip install --upgrade twine
	pip install --upgrade pip setuptools wheel
	rm -rf dist build open_metric_learning.egg-info
	python3 setup.py sdist bdist_wheel

.PHONY: upload_to_pip
upload_to_pip: build_wheel
	twine upload dist/*

.PHONY: pip_install_actual_oml
pip_install_actual_oml:
	pip install open-metric-learning==$(OML_VERSION)
