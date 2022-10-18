## Before you start
* Read our [FAQ](https://github.com/OML-Team/open-metric-learning#faq).
* Check out [python examples](https://github.com/OML-Team/open-metric-learning#get-started-using-python)
  and [config examples](https://github.com/OML-Team/open-metric-learning/tree/main/examples).

## Choosing a task
* You can work on one of the [existing issues](https://github.com/OML-Team/open-metric-learning)
  or create the new one. Especially pay attention to the issues marked with the `good_first_issue` flag.
* Start the conversation under the issue that you picked. We will discuss the design and content of the pull request and
  then you can start working on it.

## Contributing
* Fork the repository.
* Clone it locally.
* Create a branch with a name that speaks for itself.
* Set up the environment. You can install the library in dev mode via `pip install -e .`
  or [build / pull docker image](https://github.com/OML-Team/open-metric-learning#installation).
* Implement the discussed functionality, **docstrings**, and **tests** for it.
* Run tests locally via `make run_tests` or `make docker_tests` (if you use docker).
* Push the code to your forked repository.
* Create a pull request to OpenMetricLearning.

## Good to know
* If your want to change `README.md` you should go to `docs/readme`, change the desired section and then build
  readme via `make build_readme`. *So, don't change the main readme file directly, otherwise tests will fail.*
* Don't forget to update the documentation if needed. Its source is located in `docs/source`. To inspect
  it locally you should run `make html` (from `docs` folder) and then open `docs/build/html/index.html` in your
  browser.
