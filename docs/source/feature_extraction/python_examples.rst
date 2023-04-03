Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Python-API is the most flexible, but knowledge-requiring approach.
You are not limited by our project structure and you can use only that part of the functionality which you need.
You can start with fully working code snippets below that train, validate and inference the model
on a tiny dataset of
`figures <https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4?usp=sharing>`_.

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train.md

.. mdinclude:: ../../../docs/readme/examples_source/extractor/val.md

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_val_pl.md

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_val_pl_ddp.md

.. mdinclude:: ../../../docs/readme/examples_source/extractor/retrieval_usage.md

Usage with PyTorch Metric Learning
########################################

You can easily access a lot of content from
`PyTorch Metric Learning <https://github.com/KevinMusgrave/pytorch-metric-learning>`_.
The examples below are different from the basic ones only in a few lines of code:

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_with_pml.md

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_with_pml_advanced.md

To use content from PyTorch Metric Learning with our Pipelines just follow the standard
`tutorial <https://open-metric-learning.readthedocs.io/en/latest/examples/config.html#how-to-use-my-own-implementation-of-loss-model-augmentations-etc>`_
of adding custom loss.

Note, during the validation process OpenMetricLearning computes *L2* distances. Thus, when choosing a distance from PML,
we recommend you to pick `distances.LpDistance(p=2)`.
