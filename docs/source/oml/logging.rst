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

* ClearML

  .. code-block:: yaml

      ...
      logger:
          name: clearml
          args:
              project_name: "test_project"
              task_name: "test"
              offline_mode: False # if True logging is directed to a local dir
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

Take a look at the example of usage the following loggers:
`Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_,
`MLFlow <mlflow.org>`_,
`ClearML <https://clear.ml/>`_,
`Neptune <https://neptune.ai/>`_ or
`WandB <https://wandb.ai/site>`_.

.. raw:: html

    <details>
    <summary><b>See example</b></summary>

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_val_pl.md

.. raw:: html

    </details>
    <br>

Using plain Python
""""""""""""""""""

Log whatever information you want using the tool of your choice.
Just take a look at:

* Criterion (loss). Some of the built-in OML's losses have their unique additional statistics,
  which is stored in the ``last_logs`` field. See the training `example <https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html>`_.

* `RetrievalResults.visualize()  <https://open-metric-learning.readthedocs.io/en/latest/contents/retrieval.html#oml.retrieval.retrieval_results.RetrievalResults.visualize>`_

* `ViTExtractor.draw_attention() <https://open-metric-learning.readthedocs.io/en/latest/contents/models.html#oml.models.vit.vit.ViTExtractor.draw_attention>`_

* `ResnetExtractor.draw_gradcam() <https://open-metric-learning.readthedocs.io/en/latest/contents/models.html#oml.models.resnet.ResnetExtractor.draw_gradcam>`_

