Pairwise Processing
=============================

Note, this part of the library is under construction.

INTERFACES
================================================================================================
.. toctree::
   :titlesonly:

.. contents::
   :local:

IDistancesPostprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.retrieval.IDistancesPostprocessor
    :undoc-members:
    :show-inheritance:

    .. automethod:: process

IPairwiseDistanceModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.models.IPairwiseDistanceModel
    :undoc-members:
    :show-inheritance:

    .. automethod:: forward

IPairsDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IPairsDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__

IMPLEMENTATIONS
================================================================================================
.. toctree::
   :titlesonly:

.. contents::
   :local:

PairwisePostprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.retrieval.postprocessors.pairwise.PairwisePostprocessor
    :undoc-members:
    :show-inheritance:

    .. automethod:: process
    .. automethod:: inference

PairwiseEmbeddingsPostprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.retrieval.postprocessors.pairwise.PairwiseEmbeddingsPostprocessor
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: inference

PairwiseImagesPostprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.retrieval.postprocessors.pairwise.PairwiseImagesPostprocessor
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: inference

EmbeddingsSiamese
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.siamese.EmbeddingsSiamese
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward

ImagesSiamese
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.siamese.ImagesSiamese
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward

EmbeddingPairsDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.datasets.pairs.EmbeddingPairsDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__

ImagePairsDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.datasets.pairs.ImagePairsDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__
