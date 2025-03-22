Base Interfaces
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

IExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.models.IExtractor
    :undoc-members:
    :show-inheritance:

    .. autoproperty:: feat_dim
    .. automethod:: extract
    .. automethod:: from_pretrained

IPairwiseModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.models.IPairwiseModel
    :undoc-members:
    :show-inheritance:

    .. automethod:: forward
    .. automethod:: predict

IFreezable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.models.IFreezable
    :undoc-members:
    :show-inheritance:

    .. automethod:: freeze
    .. automethod:: unfreeze

IBatchSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.samplers.IBatchSampler
    :undoc-members:
    :show-inheritance:

    .. automethod:: __len__
    .. automethod:: __iter__

ITripletLossWithMiner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.criterions.ITripletLossWithMiner
    :undoc-members:
    :show-inheritance:

    .. automethod:: forward

IIndexedDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IIndexedDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __getitem__

IBaseDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IBaseDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __getitem__

ILabeledDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.ILabeledDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __getitem__
    .. automethod:: get_labels
    .. automethod:: get_label2category

IQueryGalleryDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IQueryGalleryDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: get_query_ids
    .. automethod:: get_gallery_ids

IQueryGalleryLabeledDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IQueryGalleryLabeledDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: get_query_ids
    .. automethod:: get_gallery_ids
    .. automethod:: get_labels
    .. automethod:: get_label2category

IPairDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IPairDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__

IVisualizableDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IVisualizableDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: visualize

IHTMLVisualizableDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IHTMLVisualizableDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: visualize_as_html

IBasicMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.metrics.IBasicMetric
    :undoc-members:
    :show-inheritance:

    .. automethod:: setup
    .. automethod:: update_data
    .. automethod:: compute_metrics

ITripletsMiner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.miners.ITripletsMiner
    :undoc-members:
    :show-inheritance:

    .. automethod:: sample

ITripletsMinerInBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.miners.ITripletsMinerInBatch
    :undoc-members:
    :show-inheritance:

    .. automethod:: _sample
    .. automethod:: sample

IPipelineLogger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.loggers.IPipelineLogger
    :undoc-members:
    :show-inheritance:

    .. automethod:: log_figure
    .. automethod:: log_pipeline_info

IRetrievalPostprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.retrieval.IRetrievalPostprocessor
    :undoc-members:
    :show-inheritance:

    .. automethod:: process
