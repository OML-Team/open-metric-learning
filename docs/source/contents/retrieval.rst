Retrieval & Post-processing
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
    .. automethod:: from_embeddings_qg
    .. automethod:: visualize
    .. automethod:: visualize_qg
    .. automethod:: visualize_with_functions
    .. automethod:: is_empty
    .. automethod:: deepcopy
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

ConstantThresholding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.retrieval.postprocessors.algo.ConstantThresholding
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: process

AdaptiveThresholding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.retrieval.postprocessors.algo.AdaptiveThresholding
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: process
