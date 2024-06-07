Examples
~~~~~~~~~~~~~~~~~~~~~~~~

Also see: `Retrieval & post-processing <https://open-metric-learning.readthedocs.io/en/latest/contents/retrieval.html>`_.

You can boost retrieval accuracy of your vector search by adding a pairwise model as re-ranker.
In the example below we train a siamese model to re-rank top retrieval outputs of the original model
by performing inference on pairs ``(query_i, output_j)`` where ``j=1..top_n``.

.. Example =============================================================
.. raw:: html

    <details>
    <summary>Training a pairwise model as re-ranker</summary>
    <p>

.. mdinclude:: ../../../docs/readme/examples_source/postprocessing/train_val.md

.. raw:: html

    </p>
    </details>

.. Example =============================================================
.. raw:: html

    <br>
    <details>
    <summary>Retrieval with a pairwise model as re-ranker</summary>
    <p>

.. mdinclude:: ../../../docs/readme/examples_source/postprocessing/predict.md

.. raw:: html

    </p>
    </details>
    <br>

The documentation for related classes is available via the `link <https://open-metric-learning.readthedocs.io/en/latest/contents/postprocessing.html>`_.
You can also check the corresponding
`pipeline <https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing/pairwise_postprocessing>`_
analogue.
