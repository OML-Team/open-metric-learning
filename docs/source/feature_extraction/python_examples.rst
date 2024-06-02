Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Python-API is the most flexible approach:
you are not limited by our project & config structures and you can use only the needed part of OML's functionality.
You will find code snippets below to train, validate and inference the model
on a tiny dataset of
`figures <https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4?usp=sharing>`_.
Here are more details regarding dataset
`format <https://open-metric-learning.readthedocs.io/en/latest/oml/data.html>`_.

`Schemas, explanations and tips <https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction#training>`_
illustrating the code below.

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_val_img_txt.md

.. Example =============================================================
.. raw:: html

    <br>
    <details>
    <summary>Using a trained model for retrieval</summary>
    <p>

.. mdinclude:: ../../../docs/readme/examples_source/extractor/retrieval_usage.md

.. raw:: html

    </p>
    </details>
    <br>

Usage with PyTorch Lightning
########################################

.. Example =============================================================
.. raw:: html

    <details>
    <summary>PyTorch Lightning</summary>
    <p>

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_val_pl.md

.. raw:: html

    </p>
    </details>

.. Example =============================================================
.. raw:: html

    <details>
    <summary>PyTorch Lightning: DDP</summary>
    <p>

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_val_pl_ddp.md

.. raw:: html

    </p>
    </details>

.. Example =============================================================
.. raw:: html

    <details>
    <summary>PyTorch Lightning: Deal with 2 validation loaders</summary>
    <p>

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_2loaders_val.md

.. raw:: html

    </p>
    </details>

    <br>

Usage with PyTorch Metric Learning
########################################

You can easily access a lot of content from
`PyTorch Metric Learning <https://github.com/KevinMusgrave/pytorch-metric-learning>`_.
The examples below are different from the basic ones only in a few lines of code:

.. Example =============================================================
.. raw:: html

    <details>
    <summary>Losses from PyTorch Metric Learning</summary>
    <p>

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_with_pml.md

.. raw:: html

    </p>
    </details>

.. Example =============================================================
.. raw:: html

    <details>
    <summary>Losses from PyTorch Metric Learning: advanced</summary>
    <p>

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_with_pml_advanced.md

.. raw:: html

    </p>
    </details>
    <br>

To use content from PyTorch Metric Learning (PML) with our Pipelines just follow the standard
`tutorial <https://open-metric-learning.readthedocs.io/en/latest/examples/config.html#how-to-use-my-own-implementation-of-loss-model-augmentations-etc>`_
of adding custom loss.

**Note!** During the validation process OpenMetricLearning computes *L2* distances. Thus, when choosing a distance from PML,
we recommend you to pick `distances.LpDistance(p=2)`.

Handling categories
############################
.. mdinclude:: ../../../docs/readme/examples_source/extractor/handling_categories.md


Handling sequences of photos
############################
.. mdinclude:: ../../../docs/readme/examples_source/extractor/val_with_sequence.md
