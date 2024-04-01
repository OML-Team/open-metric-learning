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

IDatasetWithLabels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IDatasetWithLabels
    :undoc-members:
    :show-inheritance:

    .. automethod:: __getitem__
    .. automethod:: get_labels

IDatasetQueryGallery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IDatasetQueryGallery
    :undoc-members:
    :show-inheritance:

    .. automethod:: __getitem__

IPairsDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IPairsDataset
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __getitem__

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
