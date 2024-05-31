Examples
~~~~~~~~~~~~~~~~~~~~~~~~

You can also boost retrieval accuracy of your features extractor by adding a postprocessor (we recommend
to check the examples above first).
In the example below we train a siamese model to re-rank top retrieval outputs of the original model
by performing inference on pairs ``(query, output_i)`` where ``i=1..top_n``.

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
