Logging & Visualization
~~~~~~~~~~~~~~~~~~~~~~~

Logging in `Pipelines <https://open-metric-learning.readthedocs.io/en/latest/oml/pipelines_general.html>`_
===========================================================================================================

There are three options to log your experiments when working with Pipelines:

* `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ — is active by default if there is no ``logger`` in config.
  It will log charts (losses, metrics, statistics) and visualisation of model's mistakes for further analysis.
  If you want to change the logging directory, you may configure the logger explicitly:

  .. code-block:: yaml

    ...
    logger:
      name: tensorboard
      args:
        save_dir: "."
    ...

* `Neptune <https://neptune.ai/>`_  — an option for advanced logging & collaboration with a team.
  It will log everything logged by Tensorboard, but also the original source code, all the configs for easier reproducibility
  and telemetry such as GPU, CPU and Memory utilization.
  Your config and run command may look like this:

  .. code-block:: yaml

    ...
    logger:
      name: neptune  # requires <NEPTUNE_API_TOKEN> as global env
      args:
        project: "oml-team/test"
    ...

  .. code-block:: bash

      export NEPTUNE_API_TOKEN=your_token; python train.py

* `Weights and Biases <https://wandb.ai/site>`_ — an option for advanced logging & collaboration with a team.
  The functionality & usage is similar to the Neptune's ones.
  Your config and run command may look like this:

  .. code-block:: yaml

      ...
      logger:
          name: wandb
          args:
            project: "test_project"
      ...

  .. code-block:: bash

      export WANDB_API_KEY=your_token; python train.py

Let's consider an example of what you get using `Neptune` for the
`feature extractor <https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction>`_
pipeline.


.. image:: https://i.ibb.co/M6VFr7b/metrics-neptune-oml.png
    :target: https://i.ibb.co/M6VFr7b/metrics-neptune-oml.png
    :width: 400
    :alt: Graphs


In the example above you can observe the graphs of:

* `Metrics <https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html>`_
  such as ``CMC@1``, ``Precision@5``, ``MAP@5``, which were provided in a config file as ``metric_args``.
  Note, you can set ``metrics_args.return_only_main_category: False``
  to log metrics independently for each of the categories (if your dataset has ones).

* Loss values averaged over batches and epochs.
  Some of the built-in OML's losses have their unique additional statistics that is also logged.
  We used
  `TripletLossWithMargin <https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html#oml.losses.triplet.TripletLossWithMiner>`_
  in our example, which comes along with tracking
  positive distances, negative distances and a fraction of active triplets (those for which loss is greater than zero).


.. image:: https://i.ibb.co/Xx4kQrB/errors-neptune-oml.png
    :target: https://i.ibb.co/Xx4kQrB/errors-neptune-oml.png
    :width: 400
    :alt: Model's mistakes


The image above shows the worst model's predictions in terms of
`MAP@5 <https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html#calc-map>`_
metric.
In particular, each row contains:

* A query (blue)
* Five closest items from a gallery to the given query & the corresponding distances (they are all red because they are irrelevant to the query)
* At most two ground truths (grey), to get an idea of what model should return

There is also the slide bar that helps to estimate your model's progress from epoch to epoch


Logging in Python
=================


Using Lightning
"""""""""""""""

The easiest is to use
`Lightning <https://github.com/Lightning-AI/lightning>`_'s
integrations with
`Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_,
`Neptune <https://neptune.ai/>`_ or
`Weights and Biases <https://wandb.ai/site>`_.

Take a look at the following example:
`Training + Validation [Lightning and logging] <https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html>`_.


Using plain Python
""""""""""""""""""

Log whatever information you want using the tool of your choice.
We just provide some tips on how to get this information.
There are two main sources of logs:

* Criterion (loss). Some of the built-in OML's losses have their unique additional statistics,
  which is stored in the ``last_logs`` field. See **Training** in the `examples <https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html>`_.

* Metrics calculator — `EmbeddingMetrics <https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html#embeddingmetrics>`_.
  It has plenty of methods useful for logging. See **Validation** in the `examples <https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html>`_.

We also recommend you take a look at:

* `Visualisation notebook <https://github.com/OML-Team/open-metric-learning/blob/main/pipelines/features_extraction/visualization.ipynb>`_
  for interactive errors analysis and visualizing attention maps.

* `ViTExtractor.draw_attention() <https://open-metric-learning.readthedocs.io/en/latest/contents/models.html#oml.models.vit.vit.ViTExtractor.draw_attention>`_

* `ResnetExtractor.draw_gradcam() <https://open-metric-learning.readthedocs.io/en/latest/contents/models.html#oml.models.resnet.ResnetExtractor.draw_gradcam>`_
