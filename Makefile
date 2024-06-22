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
	# Documentation
	echo "\n## [Documentation](https://open-metric-learning.readthedocs.io/en/latest/index.html)\n" >> ${README_FILE}
	cat docs/readme/faq.md >> ${README_FILE}
	cat docs/readme/documentation.md >> ${README_FILE}
	# Installation
	echo "\n## [Installation](https://open-metric-learning.readthedocs.io/en/latest/oml/installation.html)\n" >> ${README_FILE}
	cat docs/readme/installation.md >> ${README_FILE}
	# OML features
	cat docs/readme/library_features.md >> ${README_FILE}
	# Python examples: image + texts, train + val
	echo "\n## [Examples](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#)\n" >> ${README_FILE}
	cat docs/readme/examples_source/extractor/train_val_img_txt.md >> ${README_FILE}
	# Retrieval usage
	echo "\n### Retrieval by trained model\n" >> ${README_FILE}
	cat docs/readme/examples_source/extractor/retrieval_usage.md >> ${README_FILE}
	echo "\n### Retrieval by trained model: streaming & txt2im\n" >> ${README_FILE}
	cat docs/readme/examples_source/extractor/retrieval_usage_streaming.md >> ${README_FILE}
	# Pipelines
	echo "\n## [Pipelines](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines)\n" >> ${README_FILE}
	cat docs/readme/pipelines.md >> ${README_FILE}
	# Zoo
	echo "\n## [Zoo](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/zoo.html)\n" >>${README_FILE}
	cat docs/readme/examples_source/zoo/models_usage.md >> ${README_FILE}
	# Contributing
	echo "\n## [Contributing guide](https://open-metric-learning.readthedocs.io/en/latest/oml/contributing.html)\n" >> ${README_FILE}
	cat docs/readme/contributing.md >> ${README_FILE}
	# Acknowledgments
	echo "\n## Acknowledgments\n" >> ${README_FILE}
	cat docs/readme/acknowledgments.md >> ${README_FILE}

# ====================================== TESTS ======================================

.PHONY: wandb_login
wandb_login:
	export WANDB_API_KEY=$(WANDB_API_KEY); wandb login

.PHONY: run_all_tests
run_all_tests: wandb_login
	export PYTORCH_ENABLE_MPS_FALLBACK=1; export PYTHONPATH=.; pytest --disable-warnings -sv tests
	pytest --disable-warnings --doctest-modules --doctest-continue-on-failure -sv oml

.PHONY: run_short_tests
run_short_tests:
	export PYTORCH_ENABLE_MPS_FALLBACK=1; export PYTHONPATH=.; pytest --disable-warnings -sv -m "not long and not needs_optional_dependency" tests
	pytest --disable-warnings --doctest-modules --doctest-continue-on-failure -sv oml

.PHONY: test_converters
test_converters:
	clear
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_sop.py     --dataset_root  data/Stanford_Online_Products
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_cub.py     --dataset_root  data/CUB_200_2011
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_cub.py     --dataset_root  data/CUB_200_2011 --no_bboxes
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_cars.py    --dataset_root  data/CARS196
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_cars.py    --dataset_root  data/CARS196 --no_bboxes
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_inshop.py  --dataset_root  data/DeepFashion_InShop
	export PYTHONWARNINGS=ignore; python pipelines/datasets_converters/convert_inshop.py  --dataset_root  data/DeepFashion_InShop --no_bboxes

.PHONY: run_precommit
run_precommit:
	pre-commit install && pre-commit run -a

# ====================================== DOCKER =============================

.PHONY: docker_build
docker_build:
	DOCKER_BUILDKIT=1 docker build --build-arg RUNTIME=$(RUNTIME) -t $(IMAGE_NAME) -f ci/Dockerfile .

.PHONY: docker_all_tests
docker_all_tests:
	docker run --env WANDB_API_KEY=$(WANDB_API_KEY) --env NEPTUNE_API_TOKEN=$(NEPTUNE_API_TOKEN) -t $(IMAGE_NAME) make run_all_tests

.PHONY: docker_short_tests
docker_short_tests:
	docker run --env WANDB_API_KEY=$(WANDB_API_KEY) --env NEPTUNE_API_TOKEN=$(NEPTUNE_API_TOKEN) -t $(IMAGE_NAME) make run_short_tests

# ====================================== PIP =============================

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

# ====================================== MISC =============================
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "lightning_logs" -exec rm -r {} +
	find . -type d -name "ml-runs" -exec rm -r {} +
	find . -type d -name "logs" -exec rm -r {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -r {} +
	find . -type d -name ".hydra" -exec rm -r {} +
	find . -type d -name "*outputs*" -exec rm -r {} +
	find . -type f -name "*inference_cache.pth*" -exec rm {} +
	find . -type f -name "*.log" -exec rm {} +
	find . -type f -name "*predictions.json" -exec rm {} +
	rm -rf docs/build
	rm -rf outputs/
