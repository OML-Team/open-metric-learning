from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def load_requirements(filename: str) -> List[str]:
    with open(filename, "r") as f:
        reqs = f.read().splitlines()
    return reqs


setup(
    # technical things
    version="0.1.10",
    packages=find_packages(include="oml"),
    python_requires=">=3.8,<4.0",
    install_requires=load_requirements("ci/requirements.txt"),
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    project_urls={
        "Homepage": "https://github.com/OML-Team/open-metric-learning",
        "Bug Tracker": "https://github.com/OML-Team/open-metric-learning/issues",
    },
    license="Apache License 2.0",
)
