Datasets
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

ImageBaseDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.datasets.images.ImageBaseDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: visualize

ImageDatasetLabeled
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.datasets.images.ImageDatasetLabeled
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: get_labels

ImageDatasetQueryGalleryLabeled
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.datasets.images.ImageDatasetQueryGalleryLabeled
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: get_query_ids
    .. automethod:: get_gallery_ids

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
