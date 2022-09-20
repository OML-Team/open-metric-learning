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

    .. automethod:: extract
    .. autoproperty:: feat_dim

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

IBasicMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.metrics.IBasicMetric
    :undoc-members:
    :show-inheritance:

    .. automethod:: setup
    .. automethod:: update_data
    .. automethod:: compute_metrics

IBasicMetricDDP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.metrics.IBasicMetricDDP
    :undoc-members:
    :show-inheritance:

    .. automethod:: sync

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
