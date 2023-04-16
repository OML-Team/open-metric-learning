DDP
=============================

.. toctree::
   :titlesonly:

.. contents::
   :local:

Note, that this is an advanced section for developers or curious users.
Normally, you don't even need to know about the existence of the classes and functions below.

IMetricDDP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.interfaces.metrics.IMetricDDP
    :undoc-members:
    :show-inheritance:

    .. automethod:: sync

EmbeddingMetricsDDP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.metrics.embeddings.EmbeddingMetricsDDP
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: setup
    .. automethod:: update_data
    .. automethod:: compute_metrics
    .. automethod:: sync

ModuleDDP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.lightning.modules.ddp.ModuleDDP
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

ExtractorModuleDDP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.lightning.modules.extractor.ExtractorModuleDDP
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

MetricValCallbackDDP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.lightning.callbacks.metric.MetricValCallbackDDP
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

DDPSamplerWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: oml.ddp.patching.DDPSamplerWrapper
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: _reload

patch_dataloader_to_ddp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: oml.ddp.patching.patch_dataloader_to_ddp

sync_dicts_ddp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: oml.ddp.utils.sync_dicts_ddp
