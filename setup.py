import io
import re
from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def load_requirements(filename: str) -> List[str]:
    with open(filename, "r") as f:
        reqs = f.read().splitlines()
    return reqs


def load_version() -> str:
    version_file = Path(__file__).parent / "oml" / "__init__.py"
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


TESTS_REQUIRE = load_requirements("ci/requirements_tests.txt")
OPTIONAL_REQUIRE = load_requirements("ci/requirements_optional.txt")
LOGGERS_REQUIRE = load_requirements("ci/requirements_loggers.txt")
NLP_REQUIRE = load_requirements("ci/requirements_nlp.txt")


setup(
    # technical things
    version=load_version(),
    packages=find_packages(exclude=["ci", "docs", "pipelines", "tests*"]),
    python_requires=">=3.8,<4.0",
    install_requires=load_requirements("ci/requirements.txt"),
    extras_require={
        "tests": TESTS_REQUIRE,
        "optional": OPTIONAL_REQUIRE,
        "loggers": LOGGERS_REQUIRE,
        "nlp": NLP_REQUIRE,
        "all": OPTIONAL_REQUIRE + TESTS_REQUIRE + LOGGERS_REQUIRE + NLP_REQUIRE,
    },
    include_package_data=True,
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    # general information
    name="open-metric-learning",
    description="OML is a PyTorch-based framework to train and validate the models producing high-quality embeddings.",
    keywords=[
        "data-science",
        "computer-vision",
        "deep-learning",
        "pytorch",
        "metric-learning",
        "representation-learning",
        "pytorch-lightning",
    ],
    author="Shabanov Aleksei",
    author_email="shabanoff.aleksei@gmail.com",
    url="https://github.com/OML-Team/open-metric-learning",
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    project_urls={
        "Homepage": "https://github.com/OML-Team/open-metric-learning",
        "Bug Tracker": "https://github.com/OML-Team/open-metric-learning/issues",
    },
    license="Apache License 2.0",
)
