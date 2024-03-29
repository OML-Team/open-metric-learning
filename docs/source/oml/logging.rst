Logging & Visualization
~~~~~~~~~~~~~~~~~~~~~~~

Logging in `Pipelines <https://open-metric-learning.readthedocs.io/en/latest/oml/pipelines_general.html>`_
===========================================================================================================

There are several loggers integrated with Pipelines. You can also `use your custom logger <file:///Users/alex/Projects/open-metric-learning/docs/build/html/feature_extraction/pipelines.html#customization>`_.


* `Tensorboard <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.tensorboard.html#module-lightning.pytorch.loggers.tensorboard>`_ — is active by default if there is no ``logger`` in config.

  .. code-block:: yaml

    ...
    logger:
      name: tensorboard
      args:
        save_dir: "."
    ...

* `Neptune <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.neptune.html#module-lightning.pytorch.loggers.neptune>`_

  .. code-block:: yaml

    ...
    logger:
      name: neptune  # requires <NEPTUNE_API_TOKEN> as global env
      args:
        project: "oml-team/test"
    ...

  .. code-block:: bash

      export NEPTUNE_API_TOKEN=your_token; python train.py

* `Weights and Biases <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb>`_

  .. code-block:: yaml

      ...
      logger:
          name: wandb
          args:
            project: "test_project"
      ...

  .. code-block:: bash

      export WANDB_API_KEY=your_token; python train.py

* `MLFlow <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html>`_

  .. code-block:: yaml

      ...
      logger:
          name: mlflow
          args:
              experiment_name: "test_project"
              tracking_uri: "file:./ml-runs"  # another way: export MLFLOW_TRACKING_URI=file:./ml-runs
      ...


An example of logging via Neptune in the
`feature extractor <https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction>`_
pipeline.


.. image:: https://i.ibb.co/M6VFr7b/metrics-neptune-oml.png
    :target: https://i.ibb.co/M6VFr7b/metrics-neptune-oml.png
    :width: 400
    :alt: Graphs


So, you get:

* `Metrics <https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html>`_
  such as ``CMC@1``, ``Precision@5``, ``MAP@5``, which were provided in a config file as ``metric_args``.
  Note, you can set ``metrics_args.return_only_overall_category: False``
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

You also get some artifacts for reproducibility, such as:

* Source code
* Config
* Dataframe
* Tags


Logging in Python
=================


Using Lightning
"""""""""""""""

Take a look at the following example:
`Training + Validation [Lightning and logging] <https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html>`_.
It shows how to use each of: `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_,
`MLFlow <mlflow.org>`_,
`Neptune <https://neptune.ai/>`_ or
`WandB <https://wandb.ai/site>`_.

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
