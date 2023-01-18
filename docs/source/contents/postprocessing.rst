Pairwise Processing
=============================

Note, this part of the library is under construction.

Interfaces
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

IPairwiseModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.models.IPairwiseModel
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

Implementations
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

LinearSiamese
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.siamese.LinearSiamese
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: forward

ResNetSiamese
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.models.siamese.ResNetSiamese
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
