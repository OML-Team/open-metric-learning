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
    .. automethod:: __getitem__
    .. automethod:: visualize

ImageLabeledDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.datasets.images.ImageLabeledDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: get_labels
    .. automethod:: get_label2category
    .. automethod:: visualize

ImageQueryGalleryLabeledDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.datasets.images.ImageQueryGalleryLabeledDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: get_query_ids
    .. automethod:: get_gallery_ids
    .. automethod:: get_labels
    .. automethod:: get_label2category
    .. automethod:: visualize

ImageQueryGalleryDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.datasets.images.ImageQueryGalleryDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: get_query_ids
    .. automethod:: get_gallery_ids
    .. automethod:: visualize

PairDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.datasets.pairs.PairDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__
