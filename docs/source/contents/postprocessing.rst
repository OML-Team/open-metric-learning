Pairwise Processing
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

Note, this part of the library is under construction.


IPostprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.postprocessor.IPostprocessor
    :undoc-members:
    :show-inheritance:

    .. automethod:: process


IPairwiseDistanceModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.models.IPairwiseDistanceModel
    :undoc-members:
    :show-inheritance:

    .. automethod:: forward


PairwiseEmbeddingsPostprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.postprocessors.pairwise_embeddings.PairwiseEmbeddingsPostprocessor
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: process


SimpleSiamese
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.siamese.SimpleSiamese
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward
