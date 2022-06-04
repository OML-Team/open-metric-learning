from setuptools import find_packages, setup

setup_kwargs = {
    "name": "open-metric-learning",
    "version": "0.1.0",
    "description": "Open-source project for Metric Learning pipelines",
    "author": "Shabanov Aleksei",
    "author_email": "shabanoff.aleksei@gmail.com",
    "url": "https://github.com/OML-Team/open-metric-learning",
    "packages": find_packages(include="oml"),
    "python_requires": ">=3.8,<4.0",
}


setup(**setup_kwargs)
