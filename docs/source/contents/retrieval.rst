Retrieval Post-Processing
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
    .. automethod:: compute_from_embeddings
    .. automethod:: visualize

PairwiseReranker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.retrieval.postprocessors.pairwise.PairwiseReranker
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: process
    .. automethod:: process_raw

