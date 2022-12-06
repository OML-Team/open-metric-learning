Metrics
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

.. autoclass:: oml.metrics.embeddings.EmbeddingMetrics
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: setup
    .. automethod:: update_data
    .. automethod:: compute_metrics
    .. automethod:: visualize
    .. automethod:: get_plot_for_queries

.. autofunction:: oml.functional.metrics.calc_retrieval_metrics

.. autofunction:: oml.functional.metrics.calc_topological_metrics

.. autofunction:: oml.functional.metrics.calc_cmc

.. autofunction:: oml.functional.metrics.calc_precision

.. autofunction:: oml.functional.metrics.calc_map

.. autofunction:: oml.functional.metrics.calc_fnmr_at_fmr

.. autofunction:: oml.functional.metrics.calc_main_components_percentage
