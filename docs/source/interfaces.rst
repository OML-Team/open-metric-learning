Base Classes
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

IExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.models.IExtractor
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

    .. automethod:: extract
    .. autoproperty:: feat_dim

IBatchSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.samplers.IBatchSampler
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

    .. automethod:: __len__
    .. automethod:: __iter__

ITripletLossWithMiner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.criterions.ITripletLossWithMiner
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

    .. automethod:: forward

IDatasetWithLabels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IDatasetWithLabels
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

    .. automethod:: __getitem__
    .. automethod:: get_labels

IDatasetQueryGallery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.datasets.IDatasetQueryGallery
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

    .. automethod:: __getitem__

IBasicMetric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.metrics.IBasicMetric
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

    .. automethod:: setup
    .. automethod:: update_data
    .. automethod:: compute_metrics

ITripletsMiner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.miners.ITripletsMiner
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

    .. automethod:: sample

ITripletsMinerInBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.miners.ITripletsMinerInBatch
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

    .. automethod:: _sample
    .. automethod:: sample
