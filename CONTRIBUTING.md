## Before you start
* Read our [FAQ](https://github.com/OML-Team/open-metric-learning#faq).
* Check out [python examples](https://github.com/OML-Team/open-metric-learning#get-started-using-python)
  and [Pipelines](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines).

## Choosing a task
* Check out our [Kanban board](https://github.com/OML-Team/open-metric-learning/projects/1).
  You can work on one of the existing issues or create a new one.
* Start the conversation under the issue you picked. We will discuss the design and content of the pull request, and
  then you can start working on it.

## Contributing in general
* Fork the repository.
* Clone it locally.
* Create a branch with a name that speaks for itself.
* Set up the environment. You can install the library in dev mode via `pip install -e .`
  or build / pull [docker image](https://github.com/OML-Team/open-metric-learning#installation).
* Implement the discussed functionality, **docstrings**, and **tests** for it.
* Run tests locally using commands from `Makefile`.
* Push the code to your forked repository.
* Create a pull request to OpenMetricLearning.

## Contributing to documentation
* If you want to change `README.md` you should go to `docs/readme`, change the desired section and then build
  readme via `make build_readme`. *So, don't change the main readme file directly, otherwise tests will fail.*
* Don't forget to update the documentation if needed. Its source is located in `docs/source`. To inspect
  it locally, you should run `make html` (from `docs` folder) and then open `docs/build/html/index.html` in your
  browser.

## Contributing to models ZOO
* Add the model's implementation under `oml/models`.
* Implement `from_pretrained()` and add the corresponding [transforms](https://github.com/OML-Team/open-metric-learning/blob/f0d151ace24aaa527d0605d055529f31ad027f49/oml/registry/transforms.py#L53).
* Add the model to `oml/registry` and `oml/configs`.
* Evaluate model on 4 benchmarks and add the results into ZOO table in the main Readme.

## Contributing to pipelines
* Implement your changes in one of the pipelines (`extractor_training_pipeline`, `extractor_validation_pipeline` or others).
* Add a new test or modify an existing one under `tests/test_runs/test_pipelines`.
* If adding a new test:
  * Add config file: `tests/test_runs/test_pipelines/configs/train_or_validate_new_feature.yaml`
  * Add python script: `tests/test_runs/test_pipelines/train_or_validate_new_feature.py`
  * Add test: `tests/test_runs/test_pipelines/test_pipelines.py`

## Don't forget to update Registry
* If you want to add some new criterion, miner, model, optimizer, sampler, lr scheduler or transforms, don't forget to
  add it to the corresponding registry (see `oml.registry`) and also add a config file (see `oml.configs`).
