Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mdinclude:: ../../../docs/readme/examples_source/extractor/train_val_all_modalities.md

Retrieval by trained model
########################################

.. mdinclude:: ../../../docs/readme/examples_source/extractor/retrieval_usage.md

.. raw:: html

    <br>

Retrieval by trained model: streaming & txt2im
##############################################

.. mdinclude:: ../../../docs/readme/examples_source/extractor/retrieval_usage_streaming.md

.. raw:: html

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
`tutorial <https://open-metric-learning.readthedocs.io/en/latest/oml/pipelines_general.html#how-to-use-my-own-implementation-of-loss-extractor-etc>`_
of adding custom loss.

**Note!** During the validation process OpenMetricLearning computes *L2* distances. Thus, when choosing a distance from PML,
we recommend you to pick `distances.LpDistance(p=2)`.

Handling categories
############################
.. mdinclude:: ../../../docs/readme/examples_source/extractor/handling_categories.md


Handling sequences of photos
############################
.. mdinclude:: ../../../docs/readme/examples_source/extractor/val_with_sequence.md
