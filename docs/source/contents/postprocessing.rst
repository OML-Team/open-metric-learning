Pairwise Processing
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

Note, this part of the library is under construction.


Interfaces
=============================


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


Implementations
=============================


PairwiseEmbeddingsPostprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.postprocessor.pairwise_embeddings.PairwiseEmbeddingsPostprocessor
    :undoc-members:
    :show-inheritance:

    .. automethod:: process


SimpleSiamese
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.SimpleSiamese
    :undoc-members:
    :show-inheritance:

    .. automethod:: forward
