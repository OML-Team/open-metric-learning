Retrieval
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

RetrievalResults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.retrieval.retrieval_results.RetrievalResults
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: from_embeddings
    .. automethod:: visualize
    .. autoproperty:: n_retrieved_items
    .. autoproperty:: distances
    .. autoproperty:: retrieved_ids
    .. autoproperty:: gt_ids

PairwiseReranker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.retrieval.postprocessors.pairwise.PairwiseReranker
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: process
    .. autoproperty:: top_n
