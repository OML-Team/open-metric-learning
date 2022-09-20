Metrics
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

EmbeddingMetrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.metrics.embeddings.EmbeddingMetrics
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: setup
    .. automethod:: update_data
    .. automethod:: compute_metrics

EmbeddingMetricsDDP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.metrics.embeddings.EmbeddingMetricsDDP
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: setup
    .. automethod:: update_data
    .. automethod:: compute_metrics
    .. automethod:: sync


calc_retrieval_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: oml.functional.metrics.calc_retrieval_metrics
